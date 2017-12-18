
// Teaf/trunk/teaf(isgw)/comm/shm_bitmap_manager.cpp

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/shm.h>
#include "platform/base_error.h"
#include "utils/teaf_shm_bitmap_manager.h"

namespace bubblefs {
namespace myteaf {

Shm_Bitmap_Manager* Shm_Bitmap_Manager::instance_ = NULL;

Shm_Bitmap_Manager::~Shm_Bitmap_Manager()
{
    
}

Shm_Bitmap_Manager* Shm_Bitmap_Manager::instance(void)
{
    if (NULL == instance_)
    {
        PRINTF_INFO("Shm_Bitmap_Manager::instance()\n");
        instance_ = new Shm_Bitmap_Manager();
    }
    return instance_;
}

// 根据KEY与最大处理的uin 初始化共享内存，并把共享内存地址影射到进程空间
// 如果共享内存已经创建，则 max_uin 无效，这时需用 ipcs -m 确认大小
int Shm_Bitmap_Manager::open(key_t key, uint32 max_uin)
{
    key_t  key_handle;
    size_t size_handle;

    if ((0 == key) || (0 == max_uin))
    {
        key_handle   = PASSPORT_SHM_KEY;
        max_uin_     = MAX_UIN;
        size_handle  = PASSPORT_SHM_SIZE;
    }
    else 
    {
        key_handle   = key; 
        max_uin_     = max_uin;        
        // 每位标志一个号码状态，每个字节存贮8个号码
        size_handle  = max_uin >> 3;
    }
    
    // Req 256M share memory
    shm_addr_ = get_share_memory(key_handle, size_handle);
    if (NULL == shm_addr_)
    {
        PRINTF_ERROR( 
            "init_shm error key=0x%08x, size=%u\n",
            key_handle, 
            (unsigned)size_handle);
        
        return -1;
    }
    
    PRINTF_INFO(
        "init_shm successfull."
        "shm_addr=0x%08lx, "
        "key=0x%08x, "
        "size=%u\n",
        (uintptr_t)(shm_addr_), 
        (int)key_handle, 
        (unsigned)size_handle);
    return 0;
}

// 把共享内存地址从进程空间去影射
int Shm_Bitmap_Manager::close(void)
{
    int ret = 0;
        
    if (NULL != shm_addr_)
    {
        ret = shmdt(shm_addr_);
        if (-1 == ret)
        {
            PRINTF_ERROR(
                "shmdt() addr=0x%08lx,errno=%d\n",
                (uintptr_t)shm_addr_, errno);
        }
    }
    
    shm_addr_ = NULL;
    return ret;
}

// 获取 key= key的共享内存的地址，并映射到进程空间；
// 如这个共享内存不存在，则建立一个大小=size,key=key的共享内存，
// 并初始化所有数据为0
void * Shm_Bitmap_Manager::get_share_memory(key_t key, size_t size)
{
    void *shm_addr = NULL;
    key_t sem_key;
    int sem_id;
    sem_key = key;//ftok( pathname, key );
    PRINTF_INFO("sem_key=%u\n", sem_key);

    // 检测指定键的共享区段是否存在
    sem_id = shmget( sem_key, size, (S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP));

    // 指定键的共享区段不存在, 建立新的共享区段, 并初始化为O
    if (-1 == sem_id)
    {
        if (ENOENT == errno)
        {
            // 共享内存的 mode=0660， 当前用户与其用户组的用户可读可写 
            sem_id = shmget( sem_key, size, ((S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP)|IPC_CREAT));
            if(-1 == sem_id)
            {
                PRINTF_ERROR(
                    "shmget() key=0x%08x,errno=%d\n",
                    sem_key, errno);
                
                return (void *)NULL;
            }
        
            shm_addr = shmat( sem_id, NULL, 0 );
            if ((void *)-1 == shm_addr)
            {
                PRINTF_ERROR( 
                    "shmat() sem_id=0x%08x,errno=%d\n",
                    sem_id, errno);
                return NULL;
                
            }
            else
            {
                memset(shm_addr, 0, size);
                PRINTF_INFO( 
                    "create key=0x%08x,sem_id=%u,shm_addr=0x%08lx\n",
                    sem_key, sem_id, (uintptr_t)shm_addr);
                return shm_addr;
            }
            
        }
        else
        {
            PRINTF_ERROR(
                "shmget() key=0x%08x,errno=%d\n",
                sem_key, errno);
            return NULL;
        }
        
    }
    else
    {
        shm_addr = shmat( sem_id, NULL, 0 );
        if ((void *)-1 == shm_addr)
        {
            PRINTF_ERROR( 
                "shmat() sem_id=0x%08x,errno=%d\n",
                sem_id, errno);
            return NULL;
            
        }
        
        PRINTF_INFO( 
            "find key=0x%08x,sem_id=%u,shmat_addr=0x%08lx\n",
            sem_key, sem_id, (uintptr_t)shm_addr);
        return shm_addr;
    }
    
}


// 查询uin的bitmap值状态
int Shm_Bitmap_Manager::get_bit(uint32 uin)
{
    if (uin < max_uin_)
    {
        return handle_shm_bit_i(shm_addr_, uin, BITMAP_CHK);
    }
    else
    {
        return 0;
    }
}

// uin的bitmap值状态置1 
// 返回值为共享内存处理后的bit 值
int Shm_Bitmap_Manager::set_bit(uint32 uin)
{
    if (uin < max_uin_)
    {
        return handle_shm_bit_i(shm_addr_, uin, BITMAP_SET);
    }
    else
    {
        return 0;
    }
}

// uin的bitmap值状态置0
// 返回值为共享内存处理后的bit 值
int Shm_Bitmap_Manager::clr_bit(uint32 uin)
{
    if (uin < max_uin_)
    {
        return handle_shm_bit_i(shm_addr_, uin, BITMAP_CLR);
    }
    else
    {
        return 0;
    }
}

inline
int Shm_Bitmap_Manager::handle_shm_bit_i(void * shm_addr, uint32 uin, int flag)
{
    uint8 bit_now = 0;    
    uint32 uin_byte_addr_offset = (uin>>3);  // uin 前29位
    uint8 uin_bit = (1<<(uin & 0x07));        // uin 后3位当 字节位索引，以次节省内存
    uint8 uin_byte =0;
    unsigned char* shm_map_addr = (unsigned char*)shm_addr;
    uint8 *curr_addr = (uint8 *)(shm_map_addr + uin_byte_addr_offset);

    if ((NULL != shm_map_addr)&& ((void *)-1 != shm_map_addr))
    {
        // uin 
        uin_byte = (*(uint8 *)curr_addr);
        uint8 bit_old = (uin_byte & uin_bit) ? 1:0;

        switch (flag)
        {
        case BITMAP_CHK: // check bit
            bit_now = ((*curr_addr) & uin_bit) ? 1:0;
            break;

        case BITMAP_SET: // set bit
            (*curr_addr) = uin_byte | uin_bit;
            bit_now = ((*curr_addr) & uin_bit) ? 1:0;

            break;

        case BITMAP_CLR: //clr bit
            (*curr_addr) = uin_byte  & (~uin_bit);
            bit_now = ((*curr_addr) & uin_bit) ? 1:0;

            break;

        default:
            break;
        }

        PRINTF_INFO(
            "uin=%u,%d=>%d\n", 
            uin,
            bit_old, 
            bit_now);


    }
    else 
    {
        PRINTF_INFO(
            "shm_addr=%p, uin=%u,uin_byte=0x%02X,uin_bit=0x%02X\n", 
            shm_addr,
            uin,
            uin_byte, 
            uin_bit
            );
        return -1;
    }

    return bit_now;
}

int Shm_Bitmap_Manager::count_bit()
{
    return count_shm_set_bit_num(shm_addr_);
}

int Shm_Bitmap_Manager::count_shm_set_bit_num(void * shm_addr)
{
    uint32 num = 0;
    uint32  uin = 10000;
    for(; uin < max_uin_; uin ++)
    {
        uint32 uin_byte_addr_offset = (uin>>3);  // uin 前29位
        uint8 uin_bit = (1<<(uin & 0x07));        // uin 后3位当 字节位索引，以次节省内存
        uint8 uin_byte =0;
        unsigned char* shm_map_addr = (unsigned char*)shm_addr;
        uint8 *curr_addr = (uint8 *)(shm_map_addr + uin_byte_addr_offset);

        //uint8 bit_now = 0;
        if ((NULL != shm_map_addr)&& ((void *)-1 != shm_map_addr))
        {
            // uin 
            uin_byte = (*(uint8 *)curr_addr);
            uint8 bit_old = (uin_byte & uin_bit) ? 1:0;
            if (1 == bit_old)
            {
                num++;
            }
        }
        else 
        {
            PRINTF_ERROR( 
                "shm_addr=%p, uin=%u,uin_byte=0x%02X,uin_bit=0x%02X\n", 
                shm_addr,
                uin,
                uin_byte, 
                uin_bit
                );
            return 0;
        }
    }
    return num;
}

} // namespace myteaf
} // namespace bubblefs