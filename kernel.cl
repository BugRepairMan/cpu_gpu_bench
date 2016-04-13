__kernel void
compute(__global float *a, __read_only float b, __read_only float c) {
	int index = get_global_id(0);

	for (int i = 0; i < 1000; ++i) {
		a[index] = a[index] * b + c;
	}
}
