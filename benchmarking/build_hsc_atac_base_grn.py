"""Build a real, ATAC-informed candidate network for Track 2's CellOracle comparison --
the thing the earlier phases of this project explicitly flagged as "the actual right
answer to does CellOracle look at ATAC data," never previously built for this project.

Pipeline (all on real, downloaded/fetched data, no fabricated inputs):
1. Real ATAC peak coordinates come straight from GSE194122's own var_names
   (`chrN-start-end`, GRCh38) -- see benchmarking/README.md, Track 2.
2. Real TSS coordinates for the 10 HSC-model genes with real expression data, fetched from
   the Ensembl REST API (GRCh38, matching the ATAC peaks' build).
3. Peaks within 10kb of each gene's TSS are that gene's "nearby peaks" (52 total across
   the 10 genes) -- their real DNA sequence is fetched from the UCSC REST API (targeted
   per-region fetch, not a full genome download -- genomepy has no genome installed here).
4. A REAL, dataset-matched background: 500 other real ATAC peaks sampled at random from
   this same GSE194122 peak set (excluding the 52 target peaks), sequences also fetched
   from UCSC. This replaces `celloracle.motif_analysis.TFinfo.scan()`'s own
   `Scanner.set_background(genome=..., size=200)` step, which needs a locally installed
   reference genome -- `set_background(fname=<our real background peaks fasta>)` does the
   same FPR-calibration job without needing that download, using this dataset's own real
   peaks as the null distribution instead of a generic genome sample.
5. `celloracle.motif_analysis.scan_dna_for_motifs`, CellOracle's OWN real scanning function
   (not a hand-rolled raw-PWM-score cutoff -- an earlier version of this script did that
   and returned an implausibly dense, near-complete graph because a bare score threshold
   with no background/FPR calibration is not a rigorous motif call), restricted to our 10
   TFs' *direct*-binding motifs (via each Motif's `.factors['direct']` annotation) at
   `fpr=0.02`, CellOracle's own tutorial default.
6. TF -> target edge if a motif hit occurs in one of target's nearby peaks.

Run in celloracle_env (has gimmemotifs + celloracle + requests):
    /opt/anaconda3/envs/celloracle_env/bin/python build_hsc_atac_base_grn.py
"""
import random
import time

import pandas as pd
import requests

OUT_DIR = "data/hsc_multiome"

# (chrom, TSS position, HSC-model gene name) -- fetched from Ensembl REST, GRCh38.
TSS = {
    "GATA1": ("X", 48783122), "GATA2": ("3", 128493208), "SPI1": ("11", 47409369),
    "FLI1": ("11", 128686535), "CEBPA": ("19", 33302534), "KLF1": ("19", 12887201),
    "GFI1": ("1", 92486925), "ZFPM1": ("16", 88453208), "TAL1": ("1", 47232225),
    "JUN": ("1", 58784048),
}
GENE_MODEL_NAME = {
    "GATA1": "GATA1", "GATA2": "GATA2", "SPI1": "PU1", "FLI1": "FLI1", "CEBPA": "CEBPA",
    "KLF1": "EKLF", "GFI1": "GFI1", "ZFPM1": "FOG1", "TAL1": "SCL", "JUN": "CJUN",
}
WINDOW = 10000
N_BACKGROUND = 500
FPR = 0.02  # CellOracle's own tutorial default for TFinfo.scan()


def load_all_peaks():
    import anndata as ad
    a = ad.read_h5ad("data/hsc_multiome/multiome_BMMC_processed.h5ad", backed="r")
    atac_names = a.var_names[(a.var["feature_types"] == "ATAC").to_numpy()]
    peaks = []
    for name in atac_names:
        chrom, s, e = name.split("-")
        peaks.append((chrom.replace("chr", ""), int(s), int(e), name))
    return pd.DataFrame(peaks, columns=["chrom", "start", "end", "peak_id"])


def fetch_sequence(chrom, start, end):
    url = f"https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom=chr{chrom};start={start};end={end}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()["dna"].upper()


def main():
    from gimmemotifs.fasta import Fasta
    from gimmemotifs.motif import default_motifs
    from gimmemotifs.scanner import Scanner
    from celloracle.motif_analysis import scan_dna_for_motifs

    peaks_df = load_all_peaks()
    print(f"{len(peaks_df)} total real ATAC peaks in the dataset")

    nearby = {}
    for gene, (chrom, tss) in TSS.items():
        hits = peaks_df[(peaks_df.chrom == chrom) & (peaks_df.start < tss + WINDOW) & (peaks_df.end > tss - WINDOW)]
        nearby[gene] = list(hits.itertuples(index=False))
        print(f"  {gene}: {len(nearby[gene])} peaks within {WINDOW}bp of its real TSS")
    target_peak_ids = {p.peak_id for peaks in nearby.values() for p in peaks}

    print(f"\nFetching {len(target_peak_ids)} target peak sequences from UCSC...")
    peak_seq = {}
    for peaks in nearby.values():
        for p in peaks:
            if p.peak_id not in peak_seq:
                peak_seq[p.peak_id] = fetch_sequence(p.chrom, p.start, p.end)
                time.sleep(0.1)

    print(f"\nSampling {N_BACKGROUND} real background peaks (excluding targets) for FPR calibration...")
    random.seed(0)
    bg_candidates = peaks_df[~peaks_df.peak_id.isin(target_peak_ids)]
    bg_sample = bg_candidates.sample(n=N_BACKGROUND, random_state=0)
    bg_seq = {}
    for _, p in bg_sample.iterrows():
        bg_seq[p.peak_id] = fetch_sequence(p.chrom, p.start, p.end)
        time.sleep(0.1)
    print(f"Fetched {len(bg_seq)} real background sequences")

    bg_fasta_path = f"{OUT_DIR}/_background_peaks.fa"
    with open(bg_fasta_path, "w") as f:
        for name, seq in bg_seq.items():
            f.write(f">{name}\n{seq}\n")

    print("\nRestricting to our 10 TFs' direct-binding motifs...")
    all_motifs = default_motifs()
    tf_motifs = {tf: [m for m in all_motifs if tf in m.factors.get("direct", [])] for tf in TSS}
    # Some motifs are shared direct binders across TFs (e.g. GATA.0001 lists GATA1, GATA2,
    # AND TAL1 as direct factors) -- dedupe by id, or gimmemotifs' internal threshold table
    # (which dedupes) ends up shorter than the raw motif list, a length-mismatch crash.
    seen_ids = set()
    our_motifs = []
    for motifs in tf_motifs.values():
        for m in motifs:
            if m.id not in seen_ids:
                seen_ids.add(m.id)
                our_motifs.append(m)
    for tf, motifs in tf_motifs.items():
        print(f"  {tf}: {len(motifs)} direct motifs" + (" (NONE -- not a DNA-binding factor)" if not motifs else ""))

    print(f"\nSetting up a properly FPR-calibrated Scanner (fpr={FPR}), background = our own real peaks...")
    scanner = Scanner()
    scanner.set_motifs(our_motifs)
    scanner.set_background(fname=bg_fasta_path)
    scanner.set_threshold(fpr=FPR)

    # celloracle.motif_analysis.process_bed_file.decompose_chrstr expects underscore-joined
    # peak IDs (`chr1_111111_222222`); GSE194122's own peak IDs use hyphens
    # (`chr1-111111-222222`) -- rename just for this Fasta object, map back via peak_to_gene.
    peak_seq_co = {name.replace("-", "_"): seq for name, seq in peak_seq.items()}
    target_fasta = Fasta(fdict=peak_seq_co)
    print("Scanning target peaks with celloracle.motif_analysis.scan_dna_for_motifs...")
    result = scan_dna_for_motifs(scanner, our_motifs, target_fasta)
    print(f"{len(result)} raw motif hits across {result.seqname.nunique()} peaks")
    result.to_csv(f"{OUT_DIR}/candidate_network_atac_real_scan_result.csv", index=False)

    # motif_id -> which of our TFs it belongs to (a motif can have multiple direct factors,
    # but here we only care about factors that are in OUR panel).
    motif_to_tfs = {}
    for tf, motifs in tf_motifs.items():
        for m in motifs:
            motif_to_tfs.setdefault(m.id, []).append(tf)

    # scan_dna_for_motifs runs every seqname through celloracle's own peak_M1 (subtracts 1
    # from the start coordinate, 1-based -> 0-based BED convention) before returning results
    # -- match that here or every lookup below silently misses (this is exactly what
    # happened on the previous run: 26 real hits, 0 edges, because of this offset).
    def to_celloracle_id(peak_id):
        chrom, start, end = peak_id.replace("-", "_").split("_")
        return f"{chrom}_{int(start) - 1}_{end}"

    peak_to_gene = {to_celloracle_id(p.peak_id): gene for gene, peaks in nearby.items() for p in peaks}

    edges = set()
    for _, row in result.iterrows():
        target_gene = peak_to_gene.get(row["seqname"])
        if target_gene is None:
            continue
        for tf in motif_to_tfs.get(row["motif_id"], []):
            edges.add((GENE_MODEL_NAME[tf], GENE_MODEL_NAME[target_gene]))

    print(f"\n{len(edges)} real ATAC+motif-derived candidate edges (FPR-calibrated):")
    for s, t in sorted(edges):
        print(f"  {s} -> {t}")

    with open(f"{OUT_DIR}/candidate_network_atac_real.csv", "w") as f:
        for s, t in sorted(edges):
            f.write(f"{s},{t}\n")
    print(f"\n-> {OUT_DIR}/candidate_network_atac_real.csv")


if __name__ == "__main__":
    main()
