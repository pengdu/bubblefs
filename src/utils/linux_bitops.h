
#ifndef BUBBLEFS_UTILS_LINUX_BITOPS_H_
#define BUBBLEFS_UTILS_LINUX_BITOPS_H_

namespace bubblefs {
namespace linux {

constexpr int NBBY = 8; /* number of bits in a byte */
/* Bit map related macros. */
#define linux_setbit(a,i)     (((unsigned char *)(a))[(i)/NBBY] |= 1<<((i)%NBBY))
#define linux_clrbit(a,i)     (((unsigned char *)(a))[(i)/NBBY] &= ~(1<<((i)%NBBY)))
#define linux_isset(a,i)                                                      \
        (((const unsigned char *)(a))[(i)/NBBY] & (1<<((i)%NBBY)))
#define linux_isclr(a,i)                                                      \
        ((((const unsigned char *)(a))[(i)/NBBY] & (1<<((i)%NBBY))) == 0)
        
}  // namespace linux
}  // namespace bubblefs

#endif // BUBBLEFS_UTILS_LINUX_BITOPS_H_