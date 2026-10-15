// Unit test for write_outputs() in c_cuda/fdtd_data.h.
//
// Builds a synthetic SimData (pseudo-random u_out, random out_reorder
// permutation), calls the real write_outputs(), reads sim_outs.h5 back and
// checks every value against u_out[out_reorder[nr]*Nt + n].  With "f32" the
// float32 store (sd.u_out32) is used and the file must be float32.
//
// Compile with -DWRITE_BLOCK_BYTES=<small> to force many/partial blocks on
// tiny data.  Driven by tests/test_write_outputs.py.
//
// usage: test_write_outputs.x Nr Nt seed [f32]     (exit 0 = pass)
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <helper_funcs.h>
#include <fdtd_common.h>
#include <fdtd_data.h>

int main(int argc, char **argv) {
   if (argc < 4) { fprintf(stderr, "usage: %s Nr Nt seed [f32]\n", argv[0]); return 2; }
   int64_t Nr = atoll(argv[1]);
   int64_t Nt = atoll(argv[2]);
   unsigned seed = (unsigned)atoi(argv[3]);
   int f32 = (argc > 4 && strcmp(argv[4], "f32") == 0);

   struct SimData sd;
   memset(&sd, 0, sizeof(sd));
   sd.Nr = Nr;
   sd.Nt = Nt;
   sd.out_float32 = (int8_t)f32;
   if (f32) mymalloc((void **)&sd.u_out32, (uint64_t)Nr*(uint64_t)Nt*sizeof(float));
   else mymalloc((void **)&sd.u_out, (uint64_t)Nr*(uint64_t)Nt*sizeof(double));
   mymalloc((void **)&sd.out_reorder, (uint64_t)Nr*sizeof(int64_t));

   srand(seed);
   for (int64_t i=0; i<Nr*Nt; i++) {
      double v = ((double)rand()/RAND_MAX - 0.5)*1e3 + (double)i*1e-9; //unique-ish per element
      if (f32) sd.u_out32[i] = (float)v;
      else sd.u_out[i] = v;
   }
   for (int64_t i=0; i<Nr; i++) sd.out_reorder[i] = i;
   for (int64_t i=Nr-1; i>0; i--) { //Fisher-Yates shuffle
      int64_t j = (int64_t)(((double)rand()/((double)RAND_MAX+1.0))*(double)(i+1));
      int64_t t = sd.out_reorder[i]; sd.out_reorder[i] = sd.out_reorder[j]; sd.out_reorder[j] = t;
   }

   write_outputs(&sd);

   //read back (as double; float32 -> double is exact)
   hid_t file = H5Fopen("sim_outs.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
   assert(file >= 0);
   hid_t dset = H5Dopen(file, "u_out", H5P_DEFAULT);
   assert(dset >= 0);
   hid_t space = H5Dget_space(dset);
   hsize_t dims[2];
   int nd = H5Sget_simple_extent_dims(space, dims, NULL);
   assert(nd == 2);
   if ((int64_t)dims[0] != Nr || (int64_t)dims[1] != Nt) {
      printf("FAIL: dataset dims (%ld,%ld) != (%ld,%ld)\n", (long)dims[0], (long)dims[1], (long)Nr, (long)Nt);
      return 1;
   }
   hid_t dtype = H5Dget_type(dset);
   if (H5Tequal(dtype, f32 ? H5T_NATIVE_FLOAT : H5T_NATIVE_DOUBLE) <= 0) {
      printf("FAIL: dataset dtype is not native %s\n", f32 ? "float" : "double");
      return 1;
   }
   H5Tclose(dtype);
   double *rb;
   mymalloc((void **)&rb, (uint64_t)Nr*(uint64_t)Nt*sizeof(double));
   herr_t st = H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, rb);
   assert(st == 0);
   H5Sclose(space); H5Dclose(dset); H5Fclose(file);

   //verify ordering and values
   int64_t nbad = 0;
   for (int64_t nr=0; nr<Nr; nr++) {
      for (int64_t n=0; n<Nt; n++) {
         int64_t k = sd.out_reorder[nr]*Nt+n;
         double want = f32 ? (double)sd.u_out32[k] : sd.u_out[k];
         if (rb[nr*Nt+n] != want) {
            if (nbad < 5) printf("MISMATCH nr=%ld n=%ld got=%.17g want=%.17g\n",
                                 (long)nr, (long)n, rb[nr*Nt+n], want);
            nbad++;
         }
      }
   }
   printf("VERIFY Nr=%ld Nt=%ld: %s (%ld mismatches)\n", (long)Nr, (long)Nt, nbad==0 ? "OK" : "FAIL", (long)nbad);
   free(rb); free(sd.u_out); free(sd.u_out32); free(sd.out_reorder);
   return nbad==0 ? 0 : 1;
}
