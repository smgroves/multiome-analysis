"""CellOracle's own real-multiome tutorial pipeline, part 2: motif-scan the peaks Cicero
identified (part 1: run_cicero.R) as co-accessible with each gene's real promoter peak,
using the same FPR-calibrated gimmemotifs Scanner already validated in
build_hsc_atac_base_grn.py (background = 500 real ATAC peaks from this dataset).

Run in celloracle_env:
    /opt/anaconda3/envs/celloracle_env/bin/python build_hsc_cicero_base_grn.py
"""
import json
import random
import time

import pandas as pd
import requests

DIR = "data/hsc_multiome/direct_net"
OUT_DIR = "data/hsc_multiome"
FPR = 0.02

GENE_MODEL_NAME = {
    "GATA1": "GATA1", "GATA2": "GATA2", "SPI1": "PU1", "FLI1": "FLI1", "CEBPA": "CEBPA",
    "KLF1": "EKLF", "GFI1": "GFI1", "ZFPM1": "FOG1", "TAL1": "SCL", "JUN": "CJUN",
}


def fetch_sequence(chrom, start, end):
    url = f"https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom={chrom};start={start};end={end}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()["dna"].upper()


def main():
    from gimmemotifs.fasta import Fasta
    from gimmemotifs.motif import default_motifs
    from gimmemotifs.scanner import Scanner
    from celloracle.motif_analysis import scan_dna_for_motifs

    with open(f"{DIR}/gene_peaks_cicero.json") as f:
        gene_peaks = json.load(f)

    all_peaks = sorted({p for peaks in gene_peaks.values() for p in peaks})
    print(f"Fetching {len(all_peaks)} unique Cicero-selected peak sequences from UCSC...")
    peak_seq = {}
    for p in all_peaks:
        chrom, start, end = p.split("_")
        peak_seq[p] = fetch_sequence(chrom, int(start), int(end))
        time.sleep(0.1)

    peaks_info = pd.read_csv(f"{DIR}/peaks_info.csv")
    peaks_info["peak_id_co"] = peaks_info["peak_id"].str.replace("-", "_")
    print(f"Sampling 500 real background peaks (same convention as the earlier ATAC base GRN)...")
    random.seed(0)
    bg_candidates = peaks_info[~peaks_info.peak_id_co.isin(all_peaks)]
    bg_sample = bg_candidates.sample(n=min(500, len(bg_candidates)), random_state=0)
    bg_seq = {}
    for _, p in bg_sample.iterrows():
        bg_seq[p.peak_id_co] = fetch_sequence(f"chr{p.chrom}", int(p.start), int(p.end))
        time.sleep(0.1)
    bg_fasta_path = f"{OUT_DIR}/_cicero_background_peaks.fa"
    with open(bg_fasta_path, "w") as f:
        for name, seq in bg_seq.items():
            f.write(f">{name}\n{seq}\n")
    print(f"Fetched {len(bg_seq)} background sequences")

    all_motifs = default_motifs()
    tf_motifs = {tf: [m for m in all_motifs if tf in m.factors.get("direct", [])] for tf in GENE_MODEL_NAME}
    seen_ids = set()
    our_motifs = []
    for motifs in tf_motifs.values():
        for m in motifs:
            if m.id not in seen_ids:
                seen_ids.add(m.id)
                our_motifs.append(m)
    for tf, motifs in tf_motifs.items():
        print(f"  {tf}: {len(motifs)} direct motifs")

    print(f"\nFPR-calibrated Scanner (fpr={FPR})...")
    scanner = Scanner()
    scanner.set_motifs(our_motifs)
    scanner.set_background(fname=bg_fasta_path)
    scanner.set_threshold(fpr=FPR)

    target_fasta = Fasta(fdict=peak_seq)
    result = scan_dna_for_motifs(scanner, our_motifs, target_fasta)
    print(f"{len(result)} raw motif hits across {result.seqname.nunique()} peaks")

    def to_celloracle_id(peak_id):
        chrom, start, end = peak_id.split("_")
        return f"{chrom}_{int(start) - 1}_{end}"

    peak_to_genes = {}
    for gene, peaks in gene_peaks.items():
        for p in peaks:
            peak_to_genes.setdefault(to_celloracle_id(p), []).append(gene)

    motif_to_tfs = {}
    for tf, motifs in tf_motifs.items():
        for m in motifs:
            motif_to_tfs.setdefault(m.id, []).append(tf)

    edges = set()
    detail = []
    for _, row in result.iterrows():
        for target_gene in peak_to_genes.get(row["seqname"], []):
            for tf in motif_to_tfs.get(row["motif_id"], []):
                edges.add((GENE_MODEL_NAME[tf], GENE_MODEL_NAME[target_gene]))
                detail.append((GENE_MODEL_NAME[tf], GENE_MODEL_NAME[target_gene], row["seqname"], row["motif_id"], row["score"]))
    pd.DataFrame(detail, columns=["source", "target", "peak", "motif_id", "score"]).to_csv(
        f"{OUT_DIR}/candidate_network_cicero_real_detail.csv", index=False
    )

    print(f"\n{len(edges)} real Cicero+motif-derived candidate edges:")
    for s, t in sorted(edges):
        print(f"  {s} -> {t}")

    with open(f"{OUT_DIR}/candidate_network_cicero_real.csv", "w") as f:
        for s, t in sorted(edges):
            f.write(f"{s},{t}\n")
    print(f"\n-> {OUT_DIR}/candidate_network_cicero_real.csv")


if __name__ == "__main__":
    main()
