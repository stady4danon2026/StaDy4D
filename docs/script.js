(function () {
  // ============== THEME TOGGLE ==============
  const themeToggle = document.getElementById('themeToggle');
  const root = document.documentElement;
  const stored = localStorage.getItem('sigma-theme');
  if (stored) root.setAttribute('data-theme', stored);

  const updateThemeIcon = () => {
    const isDark = root.getAttribute('data-theme') === 'dark';
    themeToggle.innerHTML = isDark
      ? '<i class="fas fa-sun"></i>'
      : '<i class="fas fa-moon"></i>';
  };
  updateThemeIcon();

  const toggleTheme = () => {
    const isDark = root.getAttribute('data-theme') === 'dark';
    const next = isDark ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    localStorage.setItem('sigma-theme', next);
    updateThemeIcon();
  };
  themeToggle.addEventListener('click', toggleTheme);
})();

document.addEventListener('DOMContentLoaded', function () {

  // ============== HERO STATIC <-> DYNAMIC ==============
  (function heroCrossfade() {
    const staticVid = document.querySelector('.cine-video-static');
    const dynamicVid = document.querySelector('.cine-video-dynamic');
    const badge = document.getElementById('modeBadge');
    const modeText = document.getElementById('modeText');
    if (!staticVid || !dynamicVid) return;

    let isStatic = true;
    let auto = true;
    let timer;

    const apply = () => {
      if (isStatic) {
        staticVid.classList.add('is-active');
        dynamicVid.classList.remove('is-active');
        badge?.classList.remove('is-dynamic');
        if (modeText) modeText.textContent = 'STATIC';
      } else {
        dynamicVid.classList.add('is-active');
        staticVid.classList.remove('is-active');
        badge?.classList.add('is-dynamic');
        if (modeText) modeText.textContent = 'DYNAMIC';
      }
    };

    const startAuto = () => {
      timer = setInterval(() => {
        isStatic = !isStatic;
        apply();
      }, 3000);
    };

    badge?.addEventListener('click', () => {
      auto = false;
      clearInterval(timer);
      isStatic = !isStatic;
      apply();
    });

    startAuto();
  })();

  // ============== SIDE NAV ACTIVE STATE + COMPACT-AT-TOP ==============
  (function sideNav() {
    const nav = document.getElementById('sideNav');
    const hero = document.querySelector('.cine-hero');
    if (nav && hero) {
      const updateCompact = () => {
        const heroBottom = hero.getBoundingClientRect().bottom;
        nav.classList.toggle('is-compact', heroBottom > 120);
      };
      updateCompact();
      window.addEventListener('scroll', updateCompact, { passive: true });
      window.addEventListener('resize', updateCompact);
    }

    const links = document.querySelectorAll('.side-nav a');
    if (!links.length) return;
    const map = new Map();
    links.forEach(a => {
      const id = a.getAttribute('href').slice(1);
      const sec = document.getElementById(id);
      if (sec) map.set(sec, a);
    });
    const obs = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          links.forEach(l => l.classList.remove('active'));
          map.get(e.target)?.classList.add('active');
        }
      });
    }, { rootMargin: '-40% 0px -55% 0px', threshold: 0 });
    map.forEach((_, sec) => obs.observe(sec));
  })();

  // ============== REVEAL ON SCROLL ==============
  const revealObserver = new IntersectionObserver(
    (entries) => entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible'); }),
    { threshold: 0.08 }
  );
  document.querySelectorAll('.reveal').forEach(s => revealObserver.observe(s));

  // ============== ANIMATED STAT COUNTERS (replayable) ==============
  const animateCount = (el) => {
    const target = parseFloat(el.dataset.count);
    const suffix = el.dataset.suffix || '';
    const divisor = parseFloat(el.dataset.divisor || 1);
    const decimals = parseInt(el.dataset.decimals || 0);
    const duration = 1600;
    const start = performance.now();
    const tick = (now) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      const value = (target * eased) / divisor;
      el.textContent = value.toFixed(decimals) + suffix;
      if (t < 1) requestAnimationFrame(tick);
      else el.textContent = (target / divisor).toFixed(decimals) + suffix;
    };
    requestAnimationFrame(tick);
  };
  const statObserver = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        animateCount(e.target);
        statObserver.unobserve(e.target);
      }
    });
  }, { threshold: 0.5 });
  document.querySelectorAll('.stat-num').forEach(el => {
    statObserver.observe(el);
  });
  document.querySelectorAll('.stat-card').forEach(card => {
    card.addEventListener('click', () => {
      const num = card.querySelector('.stat-num');
      if (num) animateCount(num);
    });
  });

  // ============== 3D TILT ==============
  document.querySelectorAll('.tilt').forEach(el => {
    el.addEventListener('mousemove', (e) => {
      const r = el.getBoundingClientRect();
      const x = e.clientX - r.left;
      const y = e.clientY - r.top;
      const rx = ((y / r.height) - 0.5) * -6;
      const ry = ((x / r.width) - 0.5) * 6;
      el.style.transform = `perspective(900px) rotateX(${rx}deg) rotateY(${ry}deg) translateY(-3px)`;
    });
    el.addEventListener('mouseleave', () => { el.style.transform = ''; });
  });

  // ============== METHOD STAGE LINKING ==============
  document.querySelectorAll('.overview-box[data-stage]').forEach(box => {
    box.addEventListener('click', () => {
      const id = box.dataset.stage;
      const target = document.querySelector(`.method-card[data-card="${id}"]`);
      if (!target) return;
      // toggle highlight
      document.querySelectorAll('.method-card').forEach(c => c.classList.remove('is-active'));
      document.querySelectorAll('.overview-box').forEach(b => b.classList.remove('is-active'));
      target.classList.add('is-active');
      box.classList.add('is-active');
      target.scrollIntoView({ behavior: 'smooth', block: 'center' });
      setTimeout(() => target.classList.remove('is-active'), 2400);
      setTimeout(() => box.classList.remove('is-active'), 2400);
    });
  });

  // ============== GALLERY ==============
  const towns = ['Town01', 'Town02', 'Town04', 'Town05', 'Town06', 'Town03', 'Town07', 'Town10HD'];
  const townDisplay = {
    Town01: 'City Map 1', Town02: 'City Map 2', Town03: 'City Map 3', Town04: 'City Map 4',
    Town05: 'City Map 5', Town06: 'City Map 6', Town07: 'City Map 7', Town10HD: 'City Map 8'
  };
  const scenes = ['00', '01', '02', '03', '04', '05'];
  const cameraOrder = ['car_forward', 'drone_forward', 'orbit_building', 'orbit_crossroad', 'cctv', 'pedestrian'];
  const cameras = {
    'car_forward':     { id: 'cam00', label: 'Dashcam' },
    'drone_forward':   { id: 'cam01', label: 'Drone' },
    'orbit_building':  { id: 'cam02', label: 'Building' },
    'orbit_crossroad': { id: 'cam03', label: 'Intersection' },
    'cctv':            { id: 'cam04', label: 'CCTV' },
    'pedestrian':      { id: 'cam05', label: 'Pedestrian' }
  };

  let currentTown = 'Town01';
  let currentCamera = 'car_forward';

  function generateGallery() {
    const container = document.getElementById('gallery-content');
    if (!container) return;
    container.innerHTML = '';
    const cam = cameras[currentCamera];
    scenes.forEach(scene => {
      const base = `${currentTown}_${scene}_${cam.id}_${currentCamera}`;
      const card = document.createElement('div');
      card.className = 'comparison-card';
      card.innerHTML = `
        <div class="comparison-card-header">Scene ${scene}</div>
        <div class="comparison-slider-wrap">
          <img class="img-static"  src="assets/data_demo/${base}_static.gif"  alt="static"  loading="lazy">
          <img class="img-dynamic" src="assets/data_demo/${base}_dynamic.gif" alt="dynamic" loading="lazy">
          <div class="slider-divider"></div>
          <div class="slider-handle"><i class="fas fa-arrows-alt-h"></i></div>
          <span class="comparison-label left">Dynamic</span>
          <span class="comparison-label right">Static</span>
        </div>`;
      container.appendChild(card);
    });
    initSliders();
  }

  function initSliders() {
    document.querySelectorAll('.comparison-slider-wrap').forEach((wrap, idx) => {
      const setPercent = (percent) => {
        wrap.querySelector('.img-dynamic').style.clipPath = `inset(0 ${100 - percent}% 0 0)`;
        wrap.querySelector('.slider-divider').style.left = percent + '%';
        wrap.querySelector('.slider-handle').style.left = percent + '%';
      };

      let auto = true;
      const phase = idx * 0.6;
      const start = performance.now();
      const animate = (now) => {
        if (!auto) return;
        const t = (now - start) / 1000;
        const p = 50 + 40 * Math.sin(t * 0.7 + phase);
        setPercent(p);
        wrap._raf = requestAnimationFrame(animate);
      };
      wrap._raf = requestAnimationFrame(animate);

      const stopAuto = () => { auto = false; cancelAnimationFrame(wrap._raf); };
      const onMove = (e) => {
        stopAuto();
        const rect = wrap.getBoundingClientRect();
        const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
        setPercent(Math.max(0, Math.min(100, (x / rect.width) * 100)));
      };
      wrap.addEventListener('mousemove', onMove);
      wrap.addEventListener('touchmove', (e) => { e.preventDefault(); onMove(e); }, { passive: false });
    });
  }

  function setActiveTownTab() {
    document.querySelectorAll('.gallery-town-tab').forEach(t => {
      t.classList.toggle('active', t.dataset.town === currentTown);
    });
  }
  function setActiveCameraTab() {
    document.querySelectorAll('.gallery-camera-tab').forEach(t => {
      t.classList.toggle('active', t.dataset.camera === currentCamera);
    });
  }

  document.querySelectorAll('.gallery-town-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      currentTown = tab.dataset.town;
      setActiveTownTab();
      generateGallery();
    });
  });
  document.querySelectorAll('.gallery-camera-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      currentCamera = tab.dataset.camera;
      setActiveCameraTab();
      generateGallery();
    });
  });

  generateGallery();

  // ============== KEYBOARD SHORTCUTS ==============
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    const key = e.key.toLowerCase();

    if (key === 't') {
      document.getElementById('themeToggle').click();
      return;
    }

    const inGalleryView = document.getElementById('gallery-content');
    if (!inGalleryView) return;

    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      e.preventDefault();
      const idx = towns.indexOf(currentTown);
      const next = e.key === 'ArrowRight' ? (idx + 1) % towns.length : (idx - 1 + towns.length) % towns.length;
      currentTown = towns[next];
      setActiveTownTab();
      generateGallery();
    }
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      const idx = cameraOrder.indexOf(currentCamera);
      const next = e.key === 'ArrowDown' ? (idx + 1) % cameraOrder.length : (idx - 1 + cameraOrder.length) % cameraOrder.length;
      currentCamera = cameraOrder[next];
      setActiveCameraTab();
      generateGallery();
    }
  });

  // ============== INFINITE SCROLL WALL ==============
  function initScrollWall() {
    const container = document.getElementById('scroll-wall');
    if (!container) return;
    container.innerHTML = '';
    const allPairs = [];
    towns.forEach(town => {
      scenes.forEach(scene => {
        Object.entries(cameras).forEach(([camName, cam]) => {
          const base = `${town}_${scene}_${cam.id}_${camName}`;
          allPairs.push({
            dynamic: `assets/data_demo/${base}_dynamic.gif`,
            static:  `assets/data_demo/${base}_static.gif`,
            label: `${townDisplay[town] || town} · ${cam.label}`
          });
        });
      });
    });
    for (let i = allPairs.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [allPairs[i], allPairs[j]] = [allPairs[j], allPairs[i]];
    }

    const rows = 3;
    const itemsPerRow = 22;
    for (let r = 0; r < rows; r++) {
      const row = document.createElement('div');
      row.className = 'scroll-row';
      for (let i = 0; i < itemsPerRow * 2; i++) {
        const pair = allPairs[(r * itemsPerRow + i) % allPairs.length];
        const item = document.createElement('div');
        item.className = 'scroll-row-item';
        const src = r % 2 === 0 ? pair.dynamic : pair.static;
        item.innerHTML = `
          <img src="${src}" alt="${pair.label}" loading="lazy" class="lightbox-img">
          <div class="scroll-label">${pair.label}</div>`;
        row.appendChild(item);
      }
      container.appendChild(row);
    }
  }
  initScrollWall();

  // ============== LIGHTBOX ==============
  const lightbox = document.getElementById('lightbox');
  const lightboxImg = document.getElementById('lightboxImg');
  const lightboxClose = document.getElementById('lightboxClose');

  const openLightbox = (src, alt) => {
    lightboxImg.src = src;
    lightboxImg.alt = alt || '';
    lightbox.classList.add('is-open');
  };
  const closeLightbox = () => {
    lightbox.classList.remove('is-open');
    lightboxImg.src = '';
  };

  document.addEventListener('click', (e) => {
    const img = e.target.closest('.lightbox-img');
    if (img) {
      e.preventDefault();
      openLightbox(img.src, img.alt);
    } else if (e.target === lightbox || e.target === lightboxClose || e.target.closest('#lightboxClose')) {
      closeLightbox();
    }
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && lightbox.classList.contains('is-open')) closeLightbox();
  });
});

// ============== BIBTEX COPY ==============
function copyBibtex(btn) {
  const text = document.getElementById('bib').innerText;
  navigator.clipboard.writeText(text).then(() => {
    const orig = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-check"></i> Copied';
    setTimeout(() => btn.innerHTML = orig, 1600);
  });
}
