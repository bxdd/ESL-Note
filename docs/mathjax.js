window.MathJax = {
  config: ["MMLorHTML.js"],
  extensions: ["tex2jax.js", "MathMenu.js", "MathZoom.js", "TeX/AMSmath.js","TeX/AMSsymbols.js"],
  jax: ["input/TeX", "output/HTML-CSS", "output/NativeMML"],
  tex2jax: {
    inlineMath: [['$','$'], ['\\(','\\)']],
    displayMath: [['$$','$$'],['\\[','\\]']],
    processEscapes:true,
  },
  TeX: {
    equationNumbers: { 
      autoNumber: "all",
      useLabelIds: true,
    },
    unicode: {
      fonts: "STIXGeneral,'Arial Unicode MS'"
    },
  },
  "HTML-CSS": {
      linebreaks: {
        automatic: true,
      },
      scale: 90,
  },
  SVG: {
    linebreaks: {
      automatic: true,
    },
  },
  displayAlign: 'center',
  showProcessingMessages: false,
  messageStyle: 'none',
  showMathMenu: false,
};
