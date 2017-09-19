#include <stdio.h>
#include "platform/atomicops.h"

using namespace bubblefs;

int main() {
	printf("start\n");
 	int a = 0;
    int b = bdcommon::atomic_add(&a, 1);
    printf("%d %d\n", a, b);
	return 0;
}
