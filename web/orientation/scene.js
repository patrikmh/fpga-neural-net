import * as THREE from './vendor/three.module.js';

const CYAN = 0x5be6e9, PINK = 0xfa78c0;

function line(points, color, opacity = .35) {
  return new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({color, transparent: true, opacity}));
}
function pod(color) {
  const group = new THREE.Group();
  const shape = new THREE.Shape();
  shape.moveTo(1.22, 0); shape.lineTo(.28, .66); shape.lineTo(-.55, .66);
  shape.lineTo(-.9, .29); shape.lineTo(-.72, -.55); shape.lineTo(.3, -.55); shape.closePath();
  const geometry = new THREE.ExtrudeGeometry(shape, {depth: .29, bevelEnabled: true,
    bevelSize: .04, bevelThickness: .04, bevelSegments: 1, steps: 1});
  group.add(new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({color: 0x192d3b,
    metalness: .75, roughness: .35, emissive: color, emissiveIntensity: .035})));
  group.add(new THREE.LineSegments(new THREE.EdgesGeometry(geometry),
    new THREE.LineBasicMaterial({color, transparent: true, opacity: .85})));
  const panel = new THREE.Mesh(new THREE.BoxGeometry(.7, .59, .06),
    new THREE.MeshStandardMaterial({color: 0x070e17, metalness: .5, roughness: .3}));
  panel.position.set(-.22, .05, .35); group.add(panel);
  const strip = new THREE.Mesh(new THREE.BoxGeometry(.5, .045, .025),
    new THREE.MeshBasicMaterial({color}));
  strip.position.set(-.22, .05, .4); group.add(strip);
  const nose = new THREE.ConeGeometry(.15, .4, 3);
  const marker = new THREE.Mesh(nose, new THREE.MeshBasicMaterial({color}));
  marker.rotation.z = -Math.PI / 2; marker.position.set(.66, 0, .41); group.add(marker);
  for (let i = 0; i < 4; i++) {
    const fin = new THREE.Mesh(new THREE.BoxGeometry(.035, .3, .035),
      new THREE.MeshStandardMaterial({color: 0x607480, metalness: .8, roughness: .3}));
    fin.position.set(-.55 + i * .15, -.37, .35); group.add(fin);
  }
  group.userData.outline = geometry;
  return group;
}

export function createOrientationScene(canvas, onSelect) {
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-10.3, 10.3, 3.5, -3.5, .1, 100);
  camera.up.set(0, 0, 1); camera.position.set(0, -14, 11); camera.lookAt(0, 0, .05);
  const renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x080f19, 0);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.3;
  scene.add(new THREE.HemisphereLight(0xb5e9ff, 0x0a0c18, 2.6));
  const key = new THREE.DirectionalLight(0xc9f9ff, 3.2);
  key.position.set(-3, -5, 9); scene.add(key);
  const rim = new THREE.DirectionalLight(PINK, 2.5);
  rim.position.set(6, 5, 5); scene.add(rim);
  const grid = new THREE.GridHelper(25, 50, 0x213c4d, 0x142632);
  grid.rotation.x = Math.PI / 2; grid.position.z = -.65; scene.add(grid);

  const source = pod(CYAN), estimate = pod(PINK);
  const sourceAnchor = new THREE.Group(), estimateAnchor = new THREE.Group();
  sourceAnchor.position.set(-7.3, 0, .4); estimateAnchor.position.set(7.3, 0, .4);
  sourceAnchor.add(source); estimateAnchor.add(estimate); scene.add(sourceAnchor, estimateAnchor);
  estimate.visible = false;
  const ghost = new THREE.LineSegments(new THREE.EdgesGeometry(source.userData.outline),
    new THREE.LineBasicMaterial({color: CYAN, transparent: true, opacity: .17}));
  ghost.scale.setScalar(1.12); sourceAnchor.add(ghost);
  for (const x of [-7.3, 7.3]) {
    const ring = new THREE.Mesh(new THREE.TorusGeometry(1.5, .012, 6, 96),
      new THREE.MeshBasicMaterial({color: x < 0 ? CYAN : PINK, transparent: true, opacity: .25}));
    ring.position.set(x, 0, -.53); scene.add(ring);
    scene.add(line([new THREE.Vector3(x - 1.8, 0, -.55), new THREE.Vector3(x + 1.8, 0, -.55)], 0x3d6577, .3));
  }

  const chip = new THREE.Group();
  const chipBody = new THREE.Mesh(new THREE.BoxGeometry(4.5, 3.5, .32),
    new THREE.MeshStandardMaterial({color: 0x101c2a, metalness: .7, roughness: .35}));
  chip.add(chipBody);
  chip.add(new THREE.LineSegments(new THREE.EdgesGeometry(chipBody.geometry),
    new THREE.LineBasicMaterial({color: 0x4c7186, transparent: true, opacity: .8})));
  const die = new THREE.Mesh(new THREE.BoxGeometry(4, 2.96, .09),
    new THREE.MeshStandardMaterial({color: 0x07121e, metalness: .6, roughness: .3}));
  die.position.z = .22; chip.add(die);
  const pinGeometry = new THREE.BoxGeometry(.09, .29, .07);
  const pinMaterial = new THREE.MeshStandardMaterial({color: 0x617d8d, metalness: .85, roughness: .25});
  for (let side of [-1, 1]) for (let i = 0; i < 20; i++) {
    const pin = new THREE.Mesh(pinGeometry, pinMaterial);
    pin.position.set(-2.02 + i * .213, side * 1.86, -.03); chip.add(pin);
  }
  const arcPoints = [];
  for (let i = 0; i <= 80; i++) {
    const angle = -.15 + i / 80 * Math.PI * 1.6;
    arcPoints.push(new THREE.Vector3(3.05 * Math.cos(angle), 2.22 * Math.sin(angle), -.35));
  }
  scene.add(line(arcPoints, PINK, .28));
  const arrow = new THREE.Mesh(new THREE.ConeGeometry(.09, .23, 3),
    new THREE.MeshBasicMaterial({color: PINK, transparent: true, opacity: .55}));
  arrow.position.copy(arcPoints.at(-1)); arrow.rotation.z = -1; scene.add(arrow);
  scene.add(chip);

  const stateGeometry = new THREE.SphereGeometry(.066, 10, 8);
  const nodes = [];
  for (let index = 0; index < 96; index++) {
    const material = new THREE.MeshStandardMaterial({color: 0x29424f, emissive: 0x142a37,
      emissiveIntensity: .1, metalness: .4, roughness: .3});
    const node = new THREE.Mesh(stateGeometry, material);
    node.position.set(-1.77 + index % 12 * .322, -1.12 + Math.floor(index / 12) * .32, .37);
    node.userData.index = index; chip.add(node); nodes.push(node);
  }
  const selection = new THREE.Mesh(new THREE.TorusGeometry(.12, .012, 6, 24),
    new THREE.MeshBasicMaterial({color: 0xd5fbff}));
  selection.position.copy(nodes[0].position); selection.position.z += .03; chip.add(selection);
  for (let row = 0; row < 8; row++) {
    scene.add(line([new THREE.Vector3(-2.15, -1.12 + row * .32, .25),
      new THREE.Vector3(2.15, -1.12 + row * .32, .25)], 0x2e6476, .12));
  }

  const sensorNodes = [], outputNodes = [];
  const sensorColors = [0x5be6e9, 0xeac279, 0x869dff];
  for (let index = 0; index < 9; index++) {
    const position = new THREE.Vector3(-4.82 + Math.floor(index / 3) * .38, -.8 + index % 3 * .8, .25);
    const node = new THREE.Mesh(new THREE.OctahedronGeometry(.09),
      new THREE.MeshBasicMaterial({color: sensorColors[Math.floor(index / 3)], transparent: true, opacity: .45}));
    node.position.copy(position); scene.add(node); sensorNodes.push(node);
    scene.add(line([new THREE.Vector3(-6, 0, .3), position, new THREE.Vector3(-2.35, -.8 + index % 3 * .8, .2)], sensorColors[Math.floor(index / 3)], .2));
  }
  for (let index = 0; index < 4; index++) {
    const position = new THREE.Vector3(4.4, -1.05 + index * .7, .25);
    const node = new THREE.Mesh(new THREE.OctahedronGeometry(.1),
      new THREE.MeshBasicMaterial({color: PINK, transparent: true, opacity: .25}));
    node.position.copy(position); scene.add(node); outputNodes.push(node);
    scene.add(line([new THREE.Vector3(2.35, -.8 + index * .53, .2), position,
      new THREE.Vector3(6.05, 0, .3)], PINK, .22));
  }
  const sourceGoal = new THREE.Quaternion(), estimateGoal = new THREE.Quaternion();
  const particles = Array.from({length: 64}, () => {
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(.075, 8, 6),
      new THREE.MeshBasicMaterial({color: CYAN, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false}));
    mesh.visible = false; scene.add(mesh);
    return {mesh, started: 0, curve: null, fade: null};
  });
  let particleIndex = 0, previousTime = performance.now();
  function verifiedPulse(index) {
    if (reducedMotion.matches) return;
    const particle = particles[particleIndex++ % particles.length];
    particle.started = performance.now(); particle.fade = null;
    particle.curve = new THREE.CatmullRomCurve3([new THREE.Vector3(-6.3, 0, .45),
      sensorNodes[index % 9].position.clone(), new THREE.Vector3(-2.25, 0, .4),
      new THREE.Vector3(-1.2, -.8, .7), new THREE.Vector3(.1, 1.15, .7),
      new THREE.Vector3(1.55, 0, .55), outputNodes[index % 4].position.clone(),
      new THREE.Vector3(6.2, 0, .45)]);
    particle.mesh.visible = true;
  }
  function resize() {
    const width = canvas.clientWidth, height = canvas.clientHeight;
    renderer.setSize(width, height, false);
    camera.top = 10.3 * height / width; camera.bottom = -camera.top;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(canvas); resize();
  function frame(now) {
    const seconds = Math.max(0, Math.min(.1, (now - previousTime) / 1000)); previousTime = now;
    const interpolation = reducedMotion.matches ? 1 : 1 - Math.exp(-seconds * 14);
    source.quaternion.slerp(sourceGoal, interpolation);
    estimate.quaternion.slerp(estimateGoal, interpolation);
    for (const particle of particles) {
      if (!particle.mesh.visible) continue;
      const progress = Math.min(1, Math.max(0, (now - particle.started) / 850));
      const fade = particle.fade === null ? 1 : Math.max(0, 1 - (now - particle.fade) / 180);
      if (progress >= 1 || fade <= 0 || reducedMotion.matches) { particle.mesh.visible = false; continue; }
      particle.mesh.position.copy(particle.curve.getPoint(progress));
      particle.mesh.material.color.set(progress < .6 ? CYAN : PINK);
      particle.mesh.material.opacity = Math.sin(progress * Math.PI) * fade;
    }
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  function quaternion(value) { return new THREE.Quaternion(value[1], value[2], value[3], value[0]); }
  return {
    setTarget(value) { ghost.quaternion.copy(quaternion(value)); },
    sample(value, pulse = true) {
      sourceGoal.copy(quaternion(value.truth_wxyz)); estimateGoal.copy(quaternion(value.quaternion_wxyz));
      estimate.visible = true;
      value.hidden.forEach((raw, index) => {
        const magnitude = Math.min(1, Math.abs(raw / 8192));
        nodes[index].material.color.set(raw < 0 ? PINK : CYAN);
        nodes[index].material.emissive.set(raw < 0 ? PINK : CYAN);
        nodes[index].material.emissiveIntensity = .15 + magnitude * 1.7;
        nodes[index].scale.setScalar(.7 + magnitude * .65);
      });
      value.sensors.forEach((raw, index) => { sensorNodes[index].material.opacity = .4 + Math.min(1, Math.abs(raw) / (index < 3 ? 10 : index < 6 ? 6 : 45)) * .6; });
      value.raw_outputs.forEach((raw, index) => { outputNodes[index].material.opacity = .25 + Math.min(1, Math.abs(raw / 8192)) * .75; });
      if (pulse) verifiedPulse(value.index);
    },
    pause() { particles.forEach(particle => { particle.fade = performance.now(); }); },
    clear() {
      estimate.visible = false; sourceGoal.identity(); ghost.quaternion.identity();
      particles.forEach(particle => { particle.mesh.visible = false; });
      nodes.forEach(node => { node.material.color.set(0x29424f); node.material.emissive.set(0x142a37); node.material.emissiveIntensity = .1; node.scale.setScalar(1); });
      sensorNodes.forEach(node => { node.material.opacity = .45; });
      outputNodes.forEach(node => { node.material.opacity = .25; });
    },
    select(index) { selection.position.copy(nodes[index].position); selection.position.z += .03; },
    pick(event) {
      const rect = canvas.getBoundingClientRect();
      const pointer = new THREE.Vector2((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
      const raycaster = new THREE.Raycaster(); raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(nodes)[0];
      if (hit) onSelect(hit.object.userData.index);
    }
  };
}

export function createForecastComparisonScene(canvas) {
  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-5, 5, 2, -2, .1, 50);
  camera.up.set(0, 0, 1); camera.position.set(0, -9, 8); camera.lookAt(0, 0, .2);
  const renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x080f19, 0);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  scene.add(new THREE.HemisphereLight(0xb5e9ff, 0x0a0c18, 2.6));
  const light = new THREE.DirectionalLight(0xc9f9ff, 3.2);
  light.position.set(-3, -5, 9); scene.add(light);
  const issued = pod(PINK), observed = pod(CYAN);
  const issuedAnchor = new THREE.Group(), observedAnchor = new THREE.Group();
  issuedAnchor.add(issued); observedAnchor.add(observed); scene.add(issuedAnchor, observedAnchor);
  issued.visible = false; observed.visible = false;
  const grid = new THREE.GridHelper(20, 40, 0x274253, 0x172b39);
  grid.rotation.x = Math.PI / 2; grid.position.z = -.65; scene.add(grid);
  function render() { renderer.render(scene, camera); }
  function resize() {
    const width = canvas.clientWidth, height = canvas.clientHeight;
    if (!width || !height) return;
    renderer.setSize(width, height, false);
    camera.top = Math.max(1.7, 5 * height / width); camera.bottom = -camera.top;
    camera.right = camera.top * width / height; camera.left = -camera.right;
    camera.updateProjectionMatrix();
    const offset = camera.right / 2;
    issuedAnchor.position.set(-offset, 0, .3); observedAnchor.position.set(offset, 0, .3);
    render();
  }
  new ResizeObserver(resize).observe(canvas); resize();
  return {
    sample(value) {
      const [w, x, y, z] = value.quaternion_wxyz;
      issued.quaternion.set(x, y, z, w);
      const [a, b, c, d] = value.truth_wxyz;
      observed.quaternion.set(b, c, d, a);
      issued.visible = true; observed.visible = true;
      render();
    },
    clear() { issued.visible = false; observed.visible = false; render(); }
  };
}
