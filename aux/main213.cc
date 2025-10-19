// Generate Pythia events corresponding to ATLAS
// Open Data (large R jet) with cuts.
// Compile with
//mainATLAS: $(PYTHIA) $$@.cc
//#ifeq ($(FASTJET3_USE),true)
//  $(CXX) $@.cc -o $@ -w $(CXX_COMMON) $(FASTJET3_OPTS)\
//   -lfastjettools
//else
//  $(error Error: $@ requires FASTJET3)
//endif
 
#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/FastJet3.h"
#include "fastjet/tools/Filter.hh"

using namespace Pythia8;


//=========================================================================


// Delta phi in the interval [-pi,pi)
const double wrap_dphi(const double a, const double b) {
  double d = a - b;
  while (d <= - M_PI) d += 2 * M_PI;
  while (d > M_PI) d -= 2 * M_PI;
  return d;
}

// Delta R between a particle and a jet.
const double deltaR(const Particle& p, const fastjet::PseudoJet& j) {
  double dphi = wrap_dphi(p.phi(),j.phi_std());
  double deta = p.eta() - j.eta();
  return sqrt(deta*deta + dphi*dphi);
}


//=========================================================================

// Make a generator, initialize, write output.
bool generate(vector<string>& config, int& eid, ofstream& output) {
  // Generator. 
  Pythia pythia;

  // Process selection.
  for (string& s : config)
    pythia.readString(s);

  // Common settings
  pythia.readString("Next:numberShowInfo = 0");
  pythia.readString("Next:numberShowProcess = 0");
  pythia.readString("Next:numberShowEvent = 0");
  pythia.readString("Beams:eCM = 13000.");

  // If Pythia fails to initialize, exit with error.
  if (!pythia.init()) return false;

  // Set up FastJet jet finder with filter
  const double Rfat = 1.0;
  const double Rsub = 0.2;
  const double fcut = 0.05;
  fastjet::JetDefinition jetDef(fastjet::antikt_algorithm, Rfat);
  fastjet::JetDefinition subDef(fastjet::kt_algorithm, Rsub);
  fastjet::Filter trimmer(subDef, fastjet::SelectorPtFractionMin(fcut));
  std::vector <fastjet::PseudoJet> fjInputs;
  // Begin event loop. Generate event. Skip if error.
  for (int iEvent = 0; iEvent < 100000 ; ++iEvent) {
    if (!pythia.next()) continue;
    fjInputs.resize(0);
    Particle lLep;
    double llpT = 0;
    Event& event = pythia.event;
    // Find the weak boson(s). There can be up to two.
    // Should only count the ones which decayed hadronically.
    vector<Particle> bosons = {};
    for (const Particle& p : event) {
      if (abs(p.id()) == 23 || abs(p.id()) == 24) {
        int d1 = p.daughter1();
        int d2 = p.daughter2();
        // Check for hadronic decays
        int nQ = 0;
        if (d1 > 0 && d2 > 0)
          for (int i = min(d1,d2); i <= max(d1,d2); ++i)
            if (abs(event[i].id()) < 6) ++nQ;
        if (nQ > 1) bosons.push_back(p);
      }
      if (!p.isFinal()) continue;
      int id = p.idAbs();
      if (abs(p.eta()) < 2.5 &&  (id == 11 || id == 13))
        if (p.pT() > llpT) {
          lLep = p;
          llpT = p.pT();
        }
      if (abs(p.eta()) > 4.5 || p.pT() < 0.5) continue;
      if (id >= 12 && id <= 16) continue;
      fastjet::PseudoJet pTmp = p;
      fjInputs.push_back(pTmp);
    }
    // Check if leading lepton pT is at least 25
    if (llpT < 25) continue;
    // Run Fastjet algorithm
    vector <fastjet::PseudoJet> inclusiveJets;
    fastjet::ClusterSequence clustSeq(fjInputs, jetDef);
    inclusiveJets = clustSeq.inclusive_jets();
    // Trim jets and keep only jets within |eta| < 2.0
    // and with pT > 250 GeV
    vector <fastjet::PseudoJet> selJets;
    for (const auto& j : inclusiveJets) {
      fastjet::PseudoJet jt = trimmer(j);
      if (!jt.has_constituents()) continue;
      if (jt.perp() < 250.0) continue;
      if (std::abs(jt.eta()) > 2.0) continue;
      selJets.push_back(jt); 
    }
    // Check for one or more large-R jet 
    // (this corresponds with what is actually in the files, and not 
    // the docs which state exactly one large-R jet)
    if (selJets.size() < 1) continue;
    // Find the jet closest to the bosons if they decayed hadronically
    vector<int> jIndex(bosons.size(), -1);
    vector<double> DeltaRMin(bosons.size(), 999);
    for (int ij = 0, N = selJets.size(); ij < N; ++ij)
      for (int ib = 0, M = bosons.size(); ib < M; ++ib) {
        const fastjet::PseudoJet& j = selJets[ij];
        const Particle& b = bosons[ib];
        double dR = deltaR(b, j);
        if (dR < DeltaRMin[ib]) {
          jIndex[ib] = ij;
          DeltaRMin[ib] = dR;
        }
    }
    
    // Write csv
    // First the jets
    for (int i = 0, N = selJets.size(); i < N; ++i) {
      const fastjet::PseudoJet& j = selJets[i];
      bool isBoson = false;
       for (int ib = 0; ib < (int)bosons.size(); ++ib)
       if (i == jIndex[ib] && DeltaRMin[ib] < 0.6) // Impose deltaR threshold for tagging
         isBoson = true;
      output << eid << ",90," << j.perp() << "," << j.eta() << 
      "," << j.phi_std() << "," << j.e() << "," << j.m() << 
      "," << (isBoson ? "1" : "0") << endl;
    }

    // Then the leading lepton
    output << eid++ << "," << abs(lLep.id()) << "," << lLep.pT() << "," <<
      lLep.eta() << "," << lLep.phi() << "," << lLep.e() << "," << lLep.m() <<
      "," << "0" << endl;

  }
  pythia.stat();

  // Done.
  return true;
}

int main() {
  // Mode selections
  vector< vector<string>> configs;
  // WZ with Z hadronically
  vector<string> wz1 = {"WeakDoubleBoson:ffbar2ZW = on", "23:onMode = off", 
    "23:onIfAny = 1 2 3 4 5", "24:onMode = off", "24:onIfAny = 11 13", };
  configs.push_back(wz1);
  // WZ with W hadronically
  vector<string> wz2 = {"WeakDoubleBoson:ffbar2ZW = on", "24:onMode = off", 
    "24:onIfAny = 1 2 3 4 5", "23:onMode = off", "23:onIfAny = 11 13", };
  configs.push_back(wz2);
  // WW
  vector<string> ww = {"WeakDoubleBoson:ffbar2WW = on"};
  configs.push_back(ww);
  // ZZ
  vector<string> zz = {"WeakDoubleBoson:ffbar2gmZgmZ = on"};
  configs.push_back(zz);
  // W or Z (leptonically) + jet. Main background.
  vector<string> wbjet = {"WeakBosonAndParton:all = on", "24:onMode = off",
    "24:onIfAny = 11 13", "23:onMode = off", "23:onIfAny = 11 13"};
  configs.push_back(wbjet);
  // Tops
  vector<string> tops = {"Top:gg2ttbar = on", "Top:qqbar2ttbar = on"};
  configs.push_back(tops);
  // Open output file
  ofstream output("pythia.csv");
  // Print header
  output << "#event_id,pid,pt [GeV],eta,phi,E [GeV],mass [GeV],v_true" << endl;
  int eid = 1;
  for (auto c : configs) {
    if(!generate(c, eid, output)) {
      cout << "Init error!" << endl;
      return 1;
    }
  }
  output.close();

  return 0;
}
