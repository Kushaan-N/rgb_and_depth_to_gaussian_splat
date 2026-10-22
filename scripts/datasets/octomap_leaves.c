/* Dump the leaves of an OctoMap binary (.bt) file: one 8-byte record per leaf
 *   uint8 depth, uint8 state (1 = occupied, 2 = free), uint16 kx, ky, kz   (node key, OctoMap convention, 16 levels)
 * Unknown space = no leaf. Format follows octomap::OcTreeBase::readBinaryNode (2 bits per child: 01 occupied leaf,
 * 10 free leaf, 11 inner node, 00 unknown; children 0..7 depth-first) and computeChildKey.
 * Used by scripts/datasets/spires_geom.py --occupancy (the Spires benchmark filters reconstructions by the TLS
 * occupancy octree). Build (CPU node): gcc -O2 -o octomap_leaves octomap_leaves.c
 *   ./octomap_leaves gt_cloud.bt leaves.bin      (prints resolution and counts)                                   */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static FILE *in, *out;
static unsigned long n_occ = 0, n_free = 0, n_inner = 0;

static void emit(int depth, int state, const uint16_t k[3]) {
  uint8_t rec[8]; rec[0] = (uint8_t)depth; rec[1] = (uint8_t)state; memcpy(rec + 2, k, 6);
  fwrite(rec, 1, 8, out);
  if (state == 1) n_occ++; else n_free++;
}

static void child_key(const uint16_t pk[3], int pos, int depth, uint16_t ck[3]) {
  /* depth = depth of the PARENT; tree_max_val 32768 */
  unsigned off = 32768u >> (depth + 1);
  for (int a = 0; a < 3; a++) {
    if (pos & (1 << a)) ck[a] = (uint16_t)(pk[a] + off);
    else ck[a] = (uint16_t)(pk[a] - off - (off ? 0 : 1));
  }
}

static void read_node(const uint16_t key[3], int depth) {
  int c1 = fgetc(in), c2 = fgetc(in);
  if (c1 == EOF || c2 == EOF) { fprintf(stderr, "unexpected EOF\n"); exit(1); }
  int has_children[8] = {0};
  n_inner++;
  for (int i = 0; i < 8; i++) {
    int byte = i < 4 ? c1 : c2, j = i % 4;
    int b0 = (byte >> (2 * j)) & 1, b1 = (byte >> (2 * j + 1)) & 1;
    uint16_t ck[3]; child_key(key, i, depth, ck);
    if (b0 == 1 && b1 == 0) emit(depth + 1, 2, ck);
    else if (b0 == 0 && b1 == 1) emit(depth + 1, 1, ck);
    else if (b0 == 1 && b1 == 1) has_children[i] = 1;
  }
  for (int i = 0; i < 8; i++)
    if (has_children[i]) { uint16_t ck[3]; child_key(key, i, depth, ck); read_node(ck, depth + 1); }
}

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s in.bt out.bin\n", argv[0]); return 2; }
  in = fopen(argv[1], "rb"); out = fopen(argv[2], "wb");
  if (!in || !out) { perror("open"); return 1; }
  char line[1024]; double res = 0;
  while (fgets(line, sizeof line, in)) {
    if (!strncmp(line, "res", 3)) res = atof(line + 4);
    if (!strncmp(line, "data", 4)) break;
  }
  uint16_t root[3] = {32768, 32768, 32768};
  read_node(root, 0);
  fclose(out);
  printf("{\"res\": %g, \"occupied_leaves\": %lu, \"free_leaves\": %lu, \"inner\": %lu}\n", res, n_occ, n_free, n_inner);
  return 0;
}
