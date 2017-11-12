
#ifndef BUBBLEFS_UTILS_LINUX_BITOPS_H_
#define BUBBLEFS_UTILS_LINUX_BITOPS_H_

namespace bubblefs {
namespace mylinux {

constexpr int MYLINUX_NBBY = 8; /* number of bits in a byte */
/* Bit map related macros. */
#define mylinux_setbit(a,i)     (((unsigned char *)(a))[(i)/MYLINUX_NBBY] |= 1<<((i)%MYLINUX_NBBY))
#define mylinux_clrbit(a,i)     (((unsigned char *)(a))[(i)/MYLINUX_NBBY] &= ~(1<<((i)%MYLINUX_NBBY)))
#define mylinux_isset(a,i)                                                      \
        (((const unsigned char *)(a))[(i)/MYLINUX_NBBY] & (1<<((i)%MYLINUX_NBBY)))
#define mylinux_isclr(a,i)                                                      \
        ((((const unsigned char *)(a))[(i)/MYLINUX_NBBY] & (1<<((i)%MYLINUX_NBBY))) == 0)
        
}  // namespace mylinux
}  // namespace bubblefs

#endif // BUBBLEFS_UTILS_LINUX_BITOPS_H_