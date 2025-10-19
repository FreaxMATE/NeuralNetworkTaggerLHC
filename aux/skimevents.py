#!/usr/bin/env python3
import ROOT, argparse, glob, csv, sys, os
ROOT.PyConfig.IgnoreCommandLineOptions = True

def build_chain(files, treename):
    ch = ROOT.TChain(treename)
    for pat in files:
        for f in glob.glob(pat):
            ch.Add(f)
    if ch.GetEntries() == 0:
        sys.exit(f"No entries found. Check files or tree name '{treename}'.")
    return ch

def main():
    ap = argparse.ArgumentParser(description="Flatten ATLAS edu large-R jets to CSV (PyROOT).")
    ap.add_argument("inputs", nargs="+", help="ROOT files or globs")
    ap.add_argument("-o", "--out", default="jets.csv", help="Output CSV")
    ap.add_argument("-t", "--tree", default="mini", help="Tree name (default: mini)")
    ap.add_argument("--jet-pt-min", type=float, default=0.0, help="Keep jets with pt >= this (GeV)")
    ap.add_argument("--lep-pt-min", type=float, default=0.0, help="Keep leptons with pt >= this (GeV)")
    ap.add_argument("--max-events", type=int, default=None, help="Process at most N events")
    args = ap.parse_args()

    ch = build_chain(args.inputs, args.tree)

    # Header
    header = ["#event_id","pid","pt [GeV]","eta","phi","E [GeV]","mass [GeV]"]

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        n = ch.GetEntries()
        if args.max_events is not None:
            n = min(n, args.max_events)

        for i in range(n):
            ch.GetEntry(i)

            # Event information
            # Event ID
            event_id = i + 1
 
            # Jet variables
            jet_pts  = ch.largeRjet_pt
            jet_etas = ch.largeRjet_eta
            jet_phis = ch.largeRjet_phi
            jet_es   = ch.largeRjet_E
            jet_ms   = ch.largeRjet_m

            # Lepton variables
            lep_pts  = ch.lep_pt
            lep_etas = ch.lep_eta
            lep_phis = ch.lep_phi
            lep_es   = ch.lep_E
            lep_ids  = ch.lep_type
            
            # Add jets to file with pid 90.
            for j in range(jet_pts.size()):
                pt = float(jet_pts[j])/1000. # convert MeV -> GeV
                if pt < args.jet_pt_min: continue
                row = [
                    event_id, 90, pt, float(jet_etas[j]), float(jet_phis[j]), 
                    float(jet_es[j])/1000., float(jet_ms[j])/1000.
                ]
                w.writerow(row)

            # Add leptons to file.
            for j in range(lep_pts.size()):
                pt = float(lep_pts[j])/1000. # convert MeV -> GeV
                if pt < args.lep_pt_min: continue
                pdgid = int(lep_ids[j])
                # Add the lepton mass
                apdg = abs(pdgid)
                ml = 0.00051099895 if apdg == 11 else (0.1056583755 if apdg == 13 else 0.0)
                row = [
                    event_id, pdgid, pt, float(lep_etas[j]), float(lep_phis[j]), 
                    float(lep_es[j])/1000., ml
                ]
                w.writerow(row)

    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
