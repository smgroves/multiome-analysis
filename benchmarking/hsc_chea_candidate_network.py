"""Build the HSC comparison's candidate network from ChEA (real ChIP-seq-derived TF-target
data, Enrichr's ChEA_2022 library), instead of from the literal Boolean model's own rules --
see benchmarking/README.md, "Combinatorial regulatory logic: the HSC ground truth", Track 1.

Only the 11 gene NAMES from HSC.txt are used here, not their connections: ChEA is queried
for every pair of panel genes independently of what BoolODE's rules actually say, so the
resulting candidate network is a real, independently-sourced (and partly wrong, partly
incomplete) scaffold -- unlike hsc_ground_truth.py's candidate_network.csv, which is built
directly from the ground-truth rules and is therefore not a real test of structure recovery.

Source: https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=ChEA_2022
(downloaded to data/chea/ChEA_2022.gmt; real ChIP-seq TF-target library, not fabricated).
"""
import os

GMT_PATH = "data/chea/ChEA_2022.gmt"
OUT_PATH = "data/hsc_ground_truth/candidate_network_chea.csv"

# HSC.txt's own gene-name convention (matches expr_bobat.csv's columns) <- ChEA/human-genome symbol.
# EGR1 is a disclosed approximation for EgrNab (a EGR1/NAB2 protein complex in the model,
# not encoded by one gene; NAB2 has zero ChEA entries anyway, see benchmarking/README.md).
CHEA_SYMBOL_TO_MODEL_NAME = {
    "GATA1": "GATA1", "GATA2": "GATA2", "ZFPM1": "FOG1", "KLF1": "EKLF",
    "FLI1": "FLI1", "TAL1": "SCL", "CEBPA": "CEBPA", "SPI1": "PU1",
    "JUN": "CJUN", "GFI1": "GFI1", "EGR1": "EGRNAB",
}


def main():
    edges = set()
    with open(GMT_PATH) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            tf = parts[0].split(" ")[0]
            if tf not in CHEA_SYMBOL_TO_MODEL_NAME:
                continue
            for target in parts[2:]:
                if target in CHEA_SYMBOL_TO_MODEL_NAME:
                    edges.add((CHEA_SYMBOL_TO_MODEL_NAME[tf], CHEA_SYMBOL_TO_MODEL_NAME[target]))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for source, target in sorted(edges):
            f.write(f"{source},{target}\n")

    srcs = {e[0] for e in edges}
    tgts = {e[1] for e in edges}
    all_genes = set(CHEA_SYMBOL_TO_MODEL_NAME.values())
    print(f"{len(edges)} ChEA-derived candidate edges over {len(srcs | tgts)} genes -> {OUT_PATH}")
    print(f"panel genes with zero ChEA-derived incoming edges (no candidate regulators at all): {all_genes - tgts}")
    print(f"panel genes never a ChEA source: {all_genes - srcs}")


if __name__ == "__main__":
    main()
