/* ================================================================
   SMART CROP RECOMMENDATION SYSTEM — script.js v3.0
   Premium interactions: count-up, sliders, dropdown, theme
   ================================================================ */

const API_URL = 'http://127.0.0.1:8000/predict';

// ── DOM refs ──────────────────────────────────────────────────────
const themeToggle  = document.getElementById('themeToggle');
const htmlEl       = document.documentElement;
const form         = document.getElementById('predictForm');
const predictBtn   = document.getElementById('predictBtn');
const resetBtn     = document.getElementById('resetBtn');

const viewIdle     = document.getElementById('resultIdle');
const viewLoading  = document.getElementById('resultLoading');
const viewError    = document.getElementById('resultError');
const viewSuccess  = document.getElementById('resultSuccess');
const errorMsg     = document.getElementById('errorMsg');

// ── Theme toggle ──────────────────────────────────────────────────
function applyTheme(theme) {
  htmlEl.setAttribute('data-theme', theme);
  themeToggle.textContent = theme === 'dark' ? '☀️' : '🌙';
  localStorage.setItem('crop-theme', theme);
}

themeToggle.addEventListener('click', () => {
  applyTheme(htmlEl.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
});

applyTheme(localStorage.getItem('crop-theme') || 'light');

// ── Floating labels ───────────────────────────────────────────────
const textInputs = document.querySelectorAll('.field-item input[type="text"], .field-item input[type="number"]');

function checkFloating(input) {
  const wrapper = input.closest('.field-item');
  if (!wrapper) return;
  if (wrapper.classList.contains('slider-val-box')) return;

  const label = wrapper.querySelector(':scope > label');
  if (label && !wrapper.classList.contains('static-label')) {
    label.classList.toggle('active', input.value.trim() !== '');
  }
}

textInputs.forEach(input => {
  input.addEventListener('focus', () => {
    const label = input.closest('.field-item')?.querySelector(':scope > label');
    if (label && !input.closest('.field-item').classList.contains('static-label')) {
      label.classList.add('active');
    }
  });
  input.addEventListener('blur', () => checkFloating(input));
  checkFloating(input);
});

// ── Slider syncing ────────────────────────────────────────────────
function syncSlider(sliderId, numId) {
  const slider = document.getElementById(sliderId);
  const num    = document.getElementById(numId);
  if (!slider || !num) return;

  const updateFill = () => {
    const min = parseFloat(slider.min) || 0;
    const max = parseFloat(slider.max) || 100;
    const pct = ((parseFloat(slider.value) - min) / (max - min)) * 100;
    slider.style.background = `linear-gradient(to right, #22c55e ${pct}%, rgba(34,197,94,0.2) ${pct}%)`;
    num.value = slider.value;
  };

  slider.addEventListener('input', updateFill);
  num.addEventListener('input', () => {
    let v = parseFloat(num.value);
    const min = parseFloat(slider.min);
    const max = parseFloat(slider.max);
    if (!isNaN(v)) {
      v = Math.min(Math.max(v, min), max);
      slider.value = v;
      updateFill();
    }
  });

  updateFill();
}

syncSlider('rainfall',      'rainfallNum');
syncSlider('temperature',   'temperatureNum');
syncSlider('humidity',      'humidityNum');
syncSlider('nitrogenRange', 'nitrogenNum');
syncSlider('phosphorusRange','phosphorusNum');
syncSlider('potassiumRange','potassiumNum');
syncSlider('soilPh',        'soilPhNum');

// ── District dropdown ─────────────────────────────────────────────
const DISTRICTS = [
  'Angul','Balangir','Balasore','Bargarh','Bhadrak','Boudh','Cuttack',
  'Deogarh','Dhenkanal','Gajapati','Ganjam','Jagatsinghpur','Jajpur',
  'Jharsuguda','Kalahandi','Kandhamal','Kendrapara','Keonjhar','Khordha',
  'Koraput','Malkangiri','Mayurbhanj','Nabarangpur','Nayagarh','Nuapada',
  'Puri','Rayagada','Sambalpur','Subarnapur','Sundargarh'
];

const districtTrigger   = document.getElementById('districtTrigger');
const districtPanel     = document.getElementById('districtPanel');
const districtSearch    = document.getElementById('districtSearch');
const districtListEl    = document.getElementById('districtList');
const districtNoResults = document.getElementById('districtNoResults');
const districtDisplay   = document.getElementById('districtDisplay');
const districtHidden    = document.getElementById('district');
const districtContainer = document.getElementById('district-container');

function renderDistricts(filter = '') {
  districtListEl.innerHTML = '';
  const filtered = DISTRICTS.filter(d => d.toLowerCase().includes(filter.toLowerCase()));

  if (filtered.length === 0) {
    districtNoResults.classList.remove('hidden');
    return;
  }
  districtNoResults.classList.add('hidden');

  filtered.forEach(d => {
    const li = document.createElement('li');
    li.className = 'dropdown-option' + (d === districtHidden.value ? ' selected' : '');
    li.textContent = d;
    li.addEventListener('click', () => {
      districtHidden.value   = d;
      districtDisplay.textContent = d;
      closeDistrictDropdown();
      renderDistricts();
    });
    districtListEl.appendChild(li);
  });
}

function openDistrictDropdown() {
  districtPanel.classList.add('show');
  districtTrigger.classList.add('active');
  districtSearch.value = '';
  renderDistricts();
  setTimeout(() => districtSearch.focus(), 50);
}

function closeDistrictDropdown() {
  districtPanel.classList.remove('show');
  districtTrigger.classList.remove('active');
}

if (districtTrigger) {
  districtTrigger.addEventListener('click', e => {
    e.stopPropagation();
    districtPanel.classList.contains('show') ? closeDistrictDropdown() : openDistrictDropdown();
  });

  districtSearch.addEventListener('input', e => renderDistricts(e.target.value));

  districtPanel.addEventListener('click', e => e.stopPropagation());

  document.addEventListener('click', () => {
    if (districtPanel.classList.contains('show')) closeDistrictDropdown();
  });

  document.addEventListener('keydown', e => {
    if (!districtPanel.classList.contains('show')) return;
    const opts = Array.from(districtListEl.querySelectorAll('.dropdown-option'));
    const focused = districtListEl.querySelector('.focused');
    let idx = opts.indexOf(focused);

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (focused) focused.classList.remove('focused');
      idx = (idx + 1) % opts.length;
      opts[idx]?.classList.add('focused');
      opts[idx]?.scrollIntoView({ block: 'nearest' });
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (focused) focused.classList.remove('focused');
      idx = (idx - 1 + opts.length) % opts.length;
      opts[idx]?.classList.add('focused');
      opts[idx]?.scrollIntoView({ block: 'nearest' });
    } else if (e.key === 'Enter' && focused) {
      e.preventDefault();
      focused.click();
    } else if (e.key === 'Escape') {
      closeDistrictDropdown();
    }
  });

  renderDistricts();
}

// ── Helpers ───────────────────────────────────────────────────────
const CROP_EMOJIS = {
  'Rice':'🌾','Wheat':'🌿','Maize':'🌽','Sugarcane':'🎋',
  'Groundnut':'🥜','Arhar':'🫘','Black gram':'🫘',
  'Green gram':'🟢','Horse gram':'🟤','Mustard':'🌼',
  'Sesamum':'🌰','Niger':'🌻','Jute':'🌿','Ragi':'🌾'
};

const fmt = {
  currency: n => {
    const val = Number(n);
    if (isNaN(val) || val === 0) return '₹0';
    return '₹' + val.toLocaleString('en-IN', { maximumFractionDigits: 0 });
  },
  yield: n => {
    const val = Number(n);
    if (isNaN(val) || val <= 0) return 'Data Unavailable';
    return val.toFixed(2) + ' t/ha';
  }
};

function switchView(view) {
  [viewIdle, viewLoading, viewError, viewSuccess].forEach(v => v.classList.add('hidden'));
  view.classList.remove('hidden');
}

// ── Count-up animation ────────────────────────────────────────────
function animateCountUp(el, target, prefix = '₹', duration = 1200) {
  const start = performance.now();
  const update = ts => {
    const progress = Math.min((ts - start) / duration, 1);
    const ease = 1 - Math.pow(1 - progress, 3); // cubic ease-out
    const current = Math.round(ease * target);
    el.textContent = prefix + current.toLocaleString('en-IN');
    if (progress < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}

// ── Form data ─────────────────────────────────────────────────────
function getFormData() {
  const v = id => document.getElementById(id)?.value ?? '';
  const n = id => {
    const num = parseFloat(v(id));
    return isNaN(num) ? 0 : num;
  };

  return {
    District:       districtHidden.value || 'Angul',
    State:          v('state'),
    Season:         v('season'),
    Soil_Type:      v('soilType'),
    Irrigation:     v('irrigation'),
    Rainfall:       n('rainfallNum'),
    Temperature:    n('temperatureNum'),
    Humidity:       n('humidityNum'),
    Nitrogen:       n('nitrogenNum'),
    Phosphorus:     n('phosphorusNum'),
    Potassium:      n('potassiumNum'),
    Soil_pH:        n('soilPhNum'),
    Area:           n('area'),
    Fertilizer:     n('fertilizer'),
    Previous_Yield: n('prevYield'),
    Year:           n('year')
  };
}

// ── Predict & render ──────────────────────────────────────────────
async function doPredict() {
  predictBtn.disabled = true;
  predictBtn.innerHTML = '<div class="spinner"></div><span class="btn-text">Analyzing…</span>';
  switchView(viewLoading);

  try {
    const resp = await fetch(API_URL, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(getFormData())
    });

    if (!resp.ok) {
      let msg = `Server Error ${resp.status}`;
      try {
        const err = await resp.json();
        if (err.detail) {
          msg += ' — ' + (Array.isArray(err.detail)
            ? err.detail.map(d => `${d.loc.join('.')}: ${d.msg}`).join(', ')
            : err.detail);
        }
      } catch (_) {}
      throw new Error(msg);
    }

    const data = await resp.json();

    // ── Crop hero ──
    const crop = data.Best_Crop || 'Unknown';
    document.getElementById('heroIcon').textContent = CROP_EMOJIS[crop] || '🌱';
    document.getElementById('bestCropTitle').textContent = crop;

    // ── Yield ──
    document.getElementById('yieldVal').textContent = fmt.yield(data.Expected_Yield);

    // ── Profit count-up ──
    const profitEl = document.getElementById('profitVal');
    const profitRaw = Number(data.Estimated_Profit) || 0;
    profitEl.textContent = '₹0';
    animateCountUp(profitEl, profitRaw, '₹', 1400);

    // ── Top 3 ──
    const topList = document.getElementById('topList');
    topList.innerHTML = '';
    (data.Top_3_Recommendations || []).slice(0, 3).forEach((r, i) => {
      const li = document.createElement('li');
      li.className = 'top-item';
      li.style.animationDelay = `${0.05 * i}s`;
      li.innerHTML = `
        <div class="top-item-left">
          <div class="rank-badge rank-${i+1}">#${i+1}</div>
          <div class="item-info">
            <span class="item-name">${CROP_EMOJIS[r.Crop] || '🌱'} ${r.Crop}</span>
            <span class="item-yield">${fmt.yield(r.Yield)}</span>
          </div>
        </div>
        <span class="item-profit">${fmt.currency(r.Profit)}</span>
      `;
      topList.appendChild(li);
    });

    // ── Insights ──
    const insList = document.getElementById('insightList');
    insList.innerHTML = '';
    (data.Insights || []).forEach(ins => {
      const d = document.createElement('div');
      d.className = 'insight-point';
      d.innerHTML = `<span>💡</span><span>${ins}</span>`;
      insList.appendChild(d);
    });

    switchView(viewSuccess);

  } catch (err) {
    console.error(err);
    errorMsg.textContent = err.message;
    switchView(viewError);
  } finally {
    predictBtn.disabled = false;
    predictBtn.innerHTML = '<span class="btn-text">🌿 Predict Best Crop</span>';
  }
}

// ── Form events ───────────────────────────────────────────────────
form.addEventListener('submit', e => {
  e.preventDefault();
  doPredict();
});

resetBtn.addEventListener('click', () => {
  form.reset();

  // Re-sync sliders
  ['rainfall','temperature','humidity','nitrogenRange','phosphorusRange','potassiumRange','soilPh']
    .forEach(id => document.getElementById(id)?.dispatchEvent(new Event('input')));

  // Re-check floating labels
  textInputs.forEach(checkFloating);

  // Reset district dropdown
  if (districtHidden && districtDisplay) {
    districtHidden.value = 'Angul';
    districtDisplay.textContent = 'Angul';
    renderDistricts();
  }

  switchView(viewIdle);
});
