
__kernel void 
test(__global const float *a, __global float *b)
{
	int gid = get_global_id(0);
	float c;
	c = a[gid];
	for (int i = 1; i < 2e5; i++) {
		c = (int) (c * c) % 10000;
	}
	b[gid] = c;
}
