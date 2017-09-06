/**
 * Tencent is pleased to support the open source community by making Tars available.
 *
 * Copyright (C) 2016THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this file except 
 * in compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software distributed 
 * under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR 
 * CONDITIONS OF ANY KIND, either express or implied. See the License for the 
 * specific language governing permissions and limitations under the License.
 */

// Tars/cpp/util/include/util/tc_mmap.cc

#include "platform/mmap.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

namespace bubblefs {
namespace io {
  
Mmaper::Mmaper(bool bOwner)
: _bOwner(bOwner)
, _pAddr(nullptr)
, _iLength(0)
, _bCreate(false)
{
}

Mmaper::~Mmaper()
{
    if (_bOwner) {
        munmap();
    }
}

void Mmaper::Mmap(size_t length, int prot, int flags, int fd, off_t offset)
{
    if (_bOwner) {
        munmap();
    }
    _pAddr      = ::mmap(nullptr, length, prot, flags, fd, offset);
    if (_pAddr == (void*)-1) {
        _pAddr = nullptr;
        throw MmapException("[Mmaper::mmap] mmap error", errno);
    }
    _iLength    = length;
    _bCreate   = false;
}

void Mmaper::Mmap(const char *file, size_t length)
{
    assert(length > 0);
    if (_bOwner) {
        munmap();
    }
    //注意_bCreate的赋值位置:保证多线程用一个对象的时候也不会有问题
    int fd = open(file, O_CREAT|O_EXCL|O_RDWR, 0666);
    if (fd == -1) {
        if (errno != EEXIST) {
            throw MmapException("[Mmaper::mmap] fopen file '" + std::string(file) + "' error", errno);
        }
        else {
            fd = open(file, O_CREAT|O_RDWR, 0666);
            if (fd == -1) {
                throw MmapException("[Mmaper::mmap] fopen file '" + std::string(file) + "' error", errno);
            }
            _bCreate = false;
        }
    }
    else {
        _bCreate = true;
    }

    lseek(fd, length-1, SEEK_SET);
    write(fd,"\0",1);

    _pAddr = ::mmap(nullptr, length, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (_pAddr == (void*)-1) {
        _pAddr = nullptr;
        close(fd);
        throw MmapException("[Mmaper::mmap] mmap file '" + std::string(file) + "' error", errno);
    }
    _iLength = length;
    if (fd != -1) {
       close(fd); 
    }
}

void Mmaper::Munmap()
{
    if (_pAddr == nullptr) {
        return;
    }

    int ret = ::munmap(_pAddr, _iLength);
    if(ret == -1) {
        throw MmapException("[Mmaper::munmap] munmap error", errno);
    }

    _pAddr     = nullptr;
    _iLength   = 0;
    _bCreate   = false;
}

void Mmaper::Msync(bool bSync)
{
    int ret = 0;
    if (bSync) {
        ret = ::msync(_pAddr, _iLength, MS_SYNC | MS_INVALIDATE);
    }
    else {
        ret = ::msync(_pAddr, _iLength, MS_ASYNC | MS_INVALIDATE);
    }
    if (ret != 0) {
        throw MmapException("[Mmaper::msync] msync error", errno);
    }
}

} // namespace io
} // namespace bubblefs