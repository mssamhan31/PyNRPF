---
title: Home
hide:
  - navigation
  - toc
---

<section class="nrpf-landing">

<header class="nrpf-hero">
  <p class="nrpf-eyebrow">Python for Network Reverse Power Flow (PyNRPF)</p>
  <h1>Some substations record only the power magnitude, not the direction.</h1>
  <p class="nrpf-lead">Reverse power flow readings, which record electricity flowing from households
  back to the grid due to solar, look normal and pass the usual checks like missing data and
  outliers. However, they have the wrong sign and should have been read as negative. PyNRPF finds
  them and puts the sign right.</p>
  <div class="nrpf-actions">
    <a class="nrpf-button nrpf-button--primary" href="overview/">Get started</a>
    <a class="nrpf-button" href="#how-it-works">How it works</a>
    <a class="nrpf-button" href="https://github.com/mssamhan31/PyNRPF">View on GitHub</a>
  </div>
</header>

<section class="nrpf-section nrpf-split">
  <div>
    <p class="nrpf-eyebrow">The problem</p>
    <h2>When solar exports are read by electricity meters as imports</h2>
    <p>On a sunny, mild day, rooftop solar can produce more electricity than the homes and
    businesses around a substation use. The surplus flows back into the network. This is called
    reverse power flow, and a correct meter records it as negative: the day's trace dips below
    zero around midday.</p>
    <p>Many older substation meters measure only how much power flows, not which way. They store
    the export as an import, so the trace touches zero and bounces back up instead of crossing
    it. Nothing looks broken, yet every one of those readings is wrong by twice its value.</p>
  </div>
  <figure class="nrpf-figure">
    <img src="assets/fig01_sign_error.png" alt="Two charts of the same day. In the first, net load dips below zero at midday. In the second, recorded by a magnitude-only meter, it touches zero and rises again.">
    <figcaption>The same day as it should be recorded (a) and as a magnitude-only meter records it
    (b). Shaded: the hours of reverse power flow. Anonymised substation data.</figcaption>
  </figure>
</section>

<section class="nrpf-section nrpf-facts">
  <p class="nrpf-eyebrow">Key facts</p>
  <h2>A growing problem, hiding in plain sight</h2>
  <div class="nrpf-stats">
    <div class="nrpf-stat"><strong>40%+</strong><span>of free-standing Australian homes have rooftop solar<span class="nrpf-ref">[1]</span></span></div>
    <div class="nrpf-stat"><strong>30 GW</strong><span>of rooftop solar installed across Australia<span class="nrpf-ref">[1]</span></span></div>
    <div class="nrpf-stat"><strong>8 of 200+</strong><span>Ausgrid substations found storing solar exports as imports in 2023<span class="nrpf-ref">[2]</span></span></div>
    <div class="nrpf-stat"><strong>34 MW</strong><span>change in the network's minimum demand once those readings were corrected (2.3%)<span class="nrpf-ref">[2]</span></span></div>
    <div class="nrpf-stat"><strong>8 MW</strong><span>largest correction to a single substation's minimum demand<span class="nrpf-ref">[2]</span></span></div>
  </div>
  <p class="nrpf-note">Ausgrid figures from the 2023 case study in the AUPEC 2026 paper, found with the
  earlier methods described there.</p>
</section>

<section class="nrpf-section">
  <p class="nrpf-eyebrow">Why it matters</p>
  <h2>A small sign with large consequences</h2>
  <div class="nrpf-cards">
    <div class="nrpf-card">
      <h3>Reporting</h3>
      <p>Network businesses report historical demand to regulators, and those figures carry the
      error too.</p>
    </div>
    <div class="nrpf-card">
      <h3>Forecasting</h3>
      <p>Net-load forecasts learn from history. Trained on wrong signs, they learn the wrong shape
      of a sunny day.</p>
    </div>
    <div class="nrpf-card">
      <h3>Network planning and investment decisions</h3>
      <p>Networks rely on net-load forecasts for network planning, which informs their
      infrastructure investment decisions. Hidden exports make minimum demand look higher than it
      really is.</p>
    </div>
  </div>
  <p>Upgrading the meters fixes future readings, but not the years of history already
  recorded. That history needs correcting.</p>
  <h3>Not one network's problem</h3>
  <p>The limitation dates from equipment installed before rooftop solar was common, when power
  was expected to flow only from the grid to customers. The same metering limitation has been
  reported at Energy Queensland<span class="nrpf-ref">[3]</span> and SA Power
  Networks<span class="nrpf-ref">[4]</span>, and a similar meter behaviour was documented in the
  United Kingdom<span class="nrpf-ref">[5]</span>.</p>
</section>

<section class="nrpf-section" id="how-it-works">
  <p class="nrpf-eyebrow">How it works</p>
  <h2>One physical idea, three possible decisions</h2>
  <div class="nrpf-steps">
    <div class="nrpf-step">
      <h3>Look at each day on its own</h3>
      <p>PyNRPF reads one substation's fifteen-minute readings for one day, together with an
      estimate of how much solar the area generated.</p>
    </div>
    <div class="nrpf-step">
      <h3>Test a flip</h3>
      <p>It tries every midday window and asks: if the sign were flipped here, would the
      underlying demand join up smoothly with the hours either side? Real demand changes
      smoothly; a wrong sign leaves a tell-tale bump.</p>
    </div>
    <div class="nrpf-step">
      <h3>Weigh the evidence</h3>
      <p>The stronger the case for a flip, the higher the probability that the day was recorded
      wrongly.</p>
    </div>
  </div>
  <div class="nrpf-flow-arrow" aria-hidden="true">&#8595;</div>
  <div class="nrpf-outcomes">
    <div class="nrpf-outcome nrpf-outcome--correct">
      <h3>Correct automatically</h3>
      <p>Clear evidence of a wrong sign: the readings in the window are flipped.</p>
    </div>
    <div class="nrpf-outcome nrpf-outcome--review">
      <h3>Ask a person</h3>
      <p>Not sure either way: the day goes to a reviewer, with the window PyNRPF suggests.</p>
    </div>
    <div class="nrpf-outcome nrpf-outcome--keep">
      <h3>Keep as recorded</h3>
      <p>No sign of an error: the day stays exactly as it was.</p>
    </div>
  </div>
  <p class="nrpf-note">One setting decides how sure PyNRPF must be before it acts on its own, and so
  how many days go to a person.</p>
</section>

<section class="nrpf-section">
  <p class="nrpf-eyebrow">Built for practice</p>
  <h2>Made to run inside a network's data pipeline</h2>
  <div class="nrpf-cards nrpf-cards--four">
    <div class="nrpf-card">
      <h3>Original data untouched</h3>
      <p>Corrections are written alongside the recorded readings, which are never overwritten.</p>
    </div>
    <div class="nrpf-card">
      <h3>Traceable</h3>
      <p>Every result records the software version and settings that produced it, so it can be
      reproduced exactly.</p>
    </div>
    <div class="nrpf-card">
      <h3>Runs where the data lives</h3>
      <p>A Python package that works on a laptop or across a cluster, one substation at a time.</p>
    </div>
    <div class="nrpf-card">
      <h3>No training needed</h3>
      <p>It works out of the box, and a few reviewed days are enough to tune it to a new
      network.</p>
    </div>
  </div>
</section>

<section class="nrpf-section">
  <p class="nrpf-eyebrow">Research</p>
  <h2>Grounded in peer-reviewed work</h2>
  <p>PyNRPF grows out of a PhD study at UNSW Sydney in partnership with Ausgrid. A conference
  paper accepted for AUPEC 2026 identifies the problem and tests two earlier methods on the
  Ausgrid network<span class="nrpf-ref">[2]</span>. The current method is the subject of a journal
  article in preparation.</p>
  <div class="nrpf-actions">
    <a class="nrpf-button" href="papers/">Papers and citation</a>
    <a class="nrpf-button" href="https://doi.org/10.5281/zenodo.18764943">Software DOI</a>
  </div>
</section>

<section class="nrpf-section">
  <p class="nrpf-eyebrow">The team</p>
  <h2>Who made it</h2>
  <div class="nrpf-team">
    <div class="nrpf-card">
      <h3>UNSW Sydney</h3>
      <p>M. Syahman Samhan, Anna Bruce and Baran Yildiz, School of Photovoltaic and Renewable
      Energy Engineering</p>
    </div>
    <div class="nrpf-card">
      <h3>Ausgrid</h3>
      <p>Jaysson Guerrero Orbe, Navid Haghdadi and Daniel Sim, Demand Forecasting</p>
    </div>
  </div>
  <p class="nrpf-small">PyNRPF is part of M. Syahman Samhan's PhD study, supported by the University
  International Postgraduate Award (UIPA) from UNSW and top-up scholarships from Ausgrid, RACE for
  2030 and the NSW Decarbonisation Innovation Hub. It is open source under the MIT licence.</p>
</section>

<section class="nrpf-section">
  <p class="nrpf-eyebrow">References</p>
  <ol class="nrpf-references">
    <li>IEA Photovoltaic Power Systems Programme, "Australia: 2025 country update."
    <a href="https://iea-pvps.org/about-iea-pvps/members/australia/">https://iea-pvps.org/about-iea-pvps/members/australia/</a></li>
    <li>M. S. Samhan, J. Guerrero Orbe, N. Haghdadi, D. Sim, A. Bruce and B. Yildiz, "Reverse power
    flow sign correction for meter-limited substation data," accepted for the Australasian
    Universities Power Engineering Conference (AUPEC), November 2026.</li>
    <li>F. Geth, M. Vanin, W. Van Westering, T. Milford and A. Pandey, "Making distribution state
    estimation practical: challenges and opportunities," arXiv:2311.07021, 2023.
    <a href="https://arxiv.org/abs/2311.07021">https://arxiv.org/abs/2311.07021</a></li>
    <li>Australian Energy Market Operator, "South Australian under frequency load shedding, dynamic
    arming: implementation investigation," 2021.
    <a href="https://www.aemo.com.au/-/media/files/initiatives/der/2021/south-australian-ufls-dynamic-arming.pdf">aemo.com.au</a></li>
    <li>Office of Gas and Electricity Markets, "Single phase import meters operating under export
    conditions, manufacturer poll: summary of responses received December 2003," 2003.
    <a href="https://www.ofgem.gov.uk/sites/default/files/docs/2004/01/5560-manufacturer_poll_response_summary.pdf">ofgem.gov.uk</a></li>
  </ol>
</section>

</section>
