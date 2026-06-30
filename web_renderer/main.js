import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const viewport = document.querySelector('#viewport');
const stats = document.querySelector('#stats');
const status = document.querySelector('#status');
const selection = document.querySelector('#selection');
const imageModal = document.querySelector('#imageModal');
const imageModalTitle = document.querySelector('#imageModalTitle');
const sourceImagePreview = document.querySelector('#sourceImagePreview');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07140c);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio || 1);
viewport.appendChild(renderer.domElement);

const camera = new THREE.PerspectiveCamera(58, 1, 0.1, 600);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

const root = new THREE.Group();
scene.add(root);

scene.add(new THREE.HemisphereLight(0xffffff, 0x203020, 1.7));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(-20, 60, -30);
scene.add(sun);

let currentGraph = null;
let selectedPlayerId = null;
let currentCameraMode = 'tactical';
const playerObjects = new Map();
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

function setStatus(message, isError = false) {
  status.textContent = message;
  status.classList.toggle('error', isError);
}

function resize() {
  const rect = viewport.getBoundingClientRect();
  renderer.setSize(rect.width, rect.height);
  camera.aspect = rect.width / rect.height;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

function clearRoot() {
  playerObjects.clear();
  while (root.children.length) {
    const child = root.children.pop();
    child.traverse((obj) => {
      obj.geometry?.dispose?.();
      if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose?.());
      else obj.material?.dispose?.();
    });
  }
}

function getPlayerById(playerId) {
  if (!currentGraph || playerId == null) return null;
  return currentGraph.players.find((player) => player.id === playerId) || null;
}

function getObservationPlayer() {
  if (!currentGraph?.players?.length) return null;
  return getPlayerById(selectedPlayerId)
    || currentGraph.players.find((player) => player.id === currentGraph.carrier_id)
    || currentGraph.players[0];
}

function updateSelectionText() {
  const player = getPlayerById(selectedPlayerId);
  if (!player) {
    selection.textContent = '未选中球员：点击场上球员即可观察。';
    return;
  }
  selection.textContent = `当前观察：#${player.id} ${player.team_name || player.team || ''}`;
}

function refreshPlayerMarkers() {
  playerObjects.forEach((object, playerId) => {
    const marker = object.userData.marker;
    if (!marker) return;
    const player = getPlayerById(playerId);
    const color = playerId === selectedPlayerId ? 0x38bdf8 : player?.is_carrier ? 0x22ff66 : player?.accent_color || '#ffffff';
    marker.material.color.set(color);
    marker.scale.setScalar(playerId === selectedPlayerId ? 1.35 : 1.0);
  });
}

function setSelectedPlayer(playerId, focusCamera = true) {
  selectedPlayerId = playerId;
  updateSelectionText();
  refreshPlayerMarkers();
  if (focusCamera && currentCameraMode !== 'tactical') setCamera(currentCameraMode);
}

function sourceImageUrl(graph) {
  const imagePath = graph?.source?.image_path;
  if (!imagePath) return null;
  const normalized = imagePath.replaceAll('\\', '/');
  const inputIndex = normalized.lastIndexOf('/input/');
  if (inputIndex >= 0) return `/input/${normalized.slice(inputIndex + '/input/'.length)}`;
  if (normalized.startsWith('input/')) return `/${normalized}`;
  return null;
}

function showSourceImage() {
  if (!currentGraph) {
    setStatus('还没有加载场景，无法展示原图', true);
    return;
  }
  const url = sourceImageUrl(currentGraph);
  if (!url) {
    setStatus('当前 scene graph 没有可访问的原图路径', true);
    return;
  }
  sourceImagePreview.src = `${url}?t=${Date.now()}`;
  imageModalTitle.textContent = currentGraph.source?.image_path || '原图';
  imageModal.classList.remove('hidden');
}

function hideSourceImage() {
  imageModal.classList.add('hidden');
  sourceImagePreview.removeAttribute('src');
}

function v3(values) {
  return new THREE.Vector3(values[0], values[1], values[2]);
}

function addPitch(graph) {
  const length = graph.pitch.length_m;
  const width = graph.pitch.width_m;
  const grass = new THREE.Mesh(
    new THREE.PlaneGeometry(length, width, 16, 8),
    new THREE.MeshStandardMaterial({ color: 0x177a37, roughness: 0.9 })
  );
  grass.rotation.x = -Math.PI / 2;
  grass.receiveShadow = true;
  root.add(grass);

  const stripeMatA = new THREE.MeshBasicMaterial({ color: 0x1d8b41, transparent: true, opacity: 0.38 });
  for (let i = 0; i < 10; i += 2) {
    const stripe = new THREE.Mesh(new THREE.PlaneGeometry(length / 10, width), stripeMatA);
    stripe.rotation.x = -Math.PI / 2;
    stripe.position.set(-length / 2 + length * (i + 0.5) / 10, 0.006, 0);
    root.add(stripe);
  }

  const lineMat = new THREE.LineBasicMaterial({ color: 0xffffff });
  const drawLine = (points) => {
    const geometry = new THREE.BufferGeometry().setFromPoints(points.map(([x, z]) => new THREE.Vector3(x, 0.035, z)));
    root.add(new THREE.Line(geometry, lineMat));
  };
  const drawCircle = (cx, cz, radius, start = 0, end = Math.PI * 2, segments = 96) => {
    const points = [];
    for (let i = 0; i <= segments; i += 1) {
      const a = start + (end - start) * (i / segments);
      points.push(new THREE.Vector3(cx + Math.cos(a) * radius, 0.04, cz + Math.sin(a) * radius));
    }
    root.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), lineMat));
  };
  const addSpot = (x, z, radius = 0.22) => {
    const spot = new THREE.Mesh(
      new THREE.CircleGeometry(radius, 24),
      new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide })
    );
    spot.rotation.x = -Math.PI / 2;
    spot.position.set(x, 0.045, z);
    root.add(spot);
  };
  const l = length / 2, w = width / 2;
  const penaltyLength = 16.5;
  const penaltyHalfWidth = 20.16;
  const goalAreaLength = 5.5;
  const goalAreaHalfWidth = 9.16;
  const penaltySpotDistance = 11.0;
  const penaltyArcRadius = 9.15;
  const cornerArcRadius = 1.0;
  const goalHalfWidth = 3.66;
  const goalDepth = 2.0;

  drawLine([[-l, -w], [l, -w], [l, w], [-l, w], [-l, -w]]);
  drawLine([[0, -w], [0, w]]);
  drawLine([[-l + penaltyLength, -penaltyHalfWidth], [-l + penaltyLength, penaltyHalfWidth], [-l, penaltyHalfWidth], [-l, -penaltyHalfWidth], [-l + penaltyLength, -penaltyHalfWidth]]);
  drawLine([[l - penaltyLength, -penaltyHalfWidth], [l - penaltyLength, penaltyHalfWidth], [l, penaltyHalfWidth], [l, -penaltyHalfWidth], [l - penaltyLength, -penaltyHalfWidth]]);
  drawLine([[-l + goalAreaLength, -goalAreaHalfWidth], [-l + goalAreaLength, goalAreaHalfWidth], [-l, goalAreaHalfWidth], [-l, -goalAreaHalfWidth], [-l + goalAreaLength, -goalAreaHalfWidth]]);
  drawLine([[l - goalAreaLength, -goalAreaHalfWidth], [l - goalAreaLength, goalAreaHalfWidth], [l, goalAreaHalfWidth], [l, -goalAreaHalfWidth], [l - goalAreaLength, -goalAreaHalfWidth]]);
  drawLine([[-l, -goalHalfWidth], [-l - goalDepth, -goalHalfWidth], [-l - goalDepth, goalHalfWidth], [-l, goalHalfWidth], [-l, -goalHalfWidth]]);
  drawLine([[l, -goalHalfWidth], [l + goalDepth, -goalHalfWidth], [l + goalDepth, goalHalfWidth], [l, goalHalfWidth], [l, -goalHalfWidth]]);
  drawCircle(0, 0, 9.15);
  addSpot(0, 0, 0.18);
  addSpot(-l + penaltySpotDistance, 0, 0.2);
  addSpot(l - penaltySpotDistance, 0, 0.2);
  drawCircle(-l + penaltySpotDistance, 0, penaltyArcRadius, -0.92, 0.92, 36);
  drawCircle(l - penaltySpotDistance, 0, penaltyArcRadius, Math.PI - 0.92, Math.PI + 0.92, 36);
  drawCircle(-l, -w, cornerArcRadius, 0, Math.PI / 2, 18);
  drawCircle(l, -w, cornerArcRadius, Math.PI / 2, Math.PI, 18);
  drawCircle(l, w, cornerArcRadius, Math.PI, Math.PI * 1.5, 18);
  drawCircle(-l, w, cornerArcRadius, Math.PI * 1.5, Math.PI * 2, 18);
}

function makePlayer(player) {
  const group = new THREE.Group();
  const kitColor = new THREE.Color(player.kit_color || '#dddddd');
  const accent = new THREE.Color(player.accent_color || '#ffffff');
  const height = player.height_m || 1.8;
  const radius = player.radius_m || 0.32;
  const isGoalkeeper = player.team === 'goalkeeper' || (player.kit_color || '').toLowerCase() === '#7cff00';
  const skinMat = new THREE.MeshStandardMaterial({ color: 0xd7a377, roughness: 0.78 });
  const kitMat = new THREE.MeshStandardMaterial({ color: kitColor, roughness: 0.58 });
  const shortsMat = new THREE.MeshStandardMaterial({ color: kitColor.clone().multiplyScalar(0.62), roughness: 0.62 });
  const sockMat = new THREE.MeshStandardMaterial({ color: accent, roughness: 0.65 });
  const bootMat = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.5 });
  const markerMat = new THREE.MeshBasicMaterial({ color: player.is_carrier ? 0x22ff66 : accent });

  const addMesh = (geometry, material, position, rotation = [0, 0, 0]) => {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(position[0], position[1], position[2]);
    mesh.rotation.set(rotation[0], rotation[1], rotation[2]);
    group.add(mesh);
    return mesh;
  };

  const limb = (length, thickness, material, position, rotation) => addMesh(
    new THREE.CapsuleGeometry(thickness, Math.max(0.05, length - thickness * 2), 5, 8),
    material,
    position,
    rotation
  );

  const bodyHeight = height * 0.44;
  const bodyY = height * 0.57;
  addMesh(
    new THREE.BoxGeometry(radius * 1.55, bodyHeight, radius * 0.86),
    kitMat,
    [0, bodyY, 0]
  );
  addMesh(
    new THREE.BoxGeometry(radius * 1.2, height * 0.16, radius * 0.8),
    shortsMat,
    [0, height * 0.34, 0]
  );

  addMesh(
    new THREE.SphereGeometry(radius * 0.58, 14, 10),
    skinMat,
    [0, height * 0.9, 0]
  );

  const shoulderY = height * 0.67;
  const hipY = height * 0.32;
  const armSwing = player.is_carrier ? 0.34 : 0.18;
  limb(height * 0.32, radius * 0.18, kitMat, [-radius * 0.58, shoulderY, 0], [0.0, 0.0, -0.26 - armSwing]);
  limb(height * 0.32, radius * 0.18, kitMat, [radius * 0.58, shoulderY, 0], [0.0, 0.0, 0.26 + armSwing]);
  limb(height * 0.34, radius * 0.17, skinMat, [-radius * 0.78, height * 0.48, 0], [0.0, 0.0, -0.16]);
  limb(height * 0.34, radius * 0.17, skinMat, [radius * 0.78, height * 0.48, 0], [0.0, 0.0, 0.16]);

  const legLean = player.is_carrier ? 0.18 : 0.08;
  limb(height * 0.36, radius * 0.21, shortsMat, [-radius * 0.28, hipY, 0], [0.0, 0.0, legLean]);
  limb(height * 0.36, radius * 0.21, shortsMat, [radius * 0.28, hipY, 0], [0.0, 0.0, -legLean]);
  limb(height * 0.34, radius * 0.17, sockMat, [-radius * 0.24, height * 0.13, 0], [0.0, 0.0, -legLean * 0.6]);
  limb(height * 0.34, radius * 0.17, sockMat, [radius * 0.24, height * 0.13, 0], [0.0, 0.0, legLean * 0.6]);
  addMesh(new THREE.BoxGeometry(radius * 0.48, radius * 0.16, radius * 0.85), bootMat, [-radius * 0.24, 0.045, radius * 0.14]);
  addMesh(new THREE.BoxGeometry(radius * 0.48, radius * 0.16, radius * 0.85), bootMat, [radius * 0.24, 0.045, radius * 0.14]);

  if (isGoalkeeper) {
    addMesh(new THREE.BoxGeometry(radius * 1.95, radius * 0.16, radius * 1.0), markerMat, [0, height * 0.78, 0]);
  }

  const labelCanvas = document.createElement('canvas');
  labelCanvas.width = 96;
  labelCanvas.height = 64;
  const ctx = labelCanvas.getContext('2d');
  ctx.fillStyle = 'rgba(0,0,0,0.66)';
  ctx.fillRect(8, 8, 80, 48);
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 34px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(String(player.id), 48, 34);
  const labelTexture = new THREE.CanvasTexture(labelCanvas);
  const label = new THREE.Sprite(new THREE.SpriteMaterial({ map: labelTexture, transparent: true }));
  label.scale.set(0.55, 0.36, 1);
  label.position.set(0, height * 1.12, 0);
  group.add(label);

  const marker = new THREE.Mesh(
    new THREE.TorusGeometry(radius * 1.55, 0.035, 8, 32),
    markerMat
  );
  marker.rotation.x = Math.PI / 2;
  marker.position.y = 0.06;
  group.add(marker);
  group.userData.marker = marker;

  const pos = v3(player.engines.three.position);
  group.position.copy(pos);
  const dir = v3(player.direction.three);
  if (dir.lengthSq() > 0.0001) group.lookAt(pos.clone().add(dir));
  group.userData.playerId = player.id;
  group.traverse((object) => {
    object.userData.playerId = player.id;
  });
  return group;
}

function addBall(graph) {
  if (!graph.ball) return;
  const ball = new THREE.Mesh(
    new THREE.SphereGeometry(graph.ball.radius_m || 0.11, 18, 12),
    new THREE.MeshStandardMaterial({ color: 0xf8fafc, roughness: 0.4 })
  );
  ball.position.copy(v3(graph.ball.engines.three.position));
  ball.position.y = Math.max(ball.position.y, (graph.ball.radius_m || 0.11));
  root.add(ball);
}

function renderGraph(graph) {
  currentGraph = graph;
  selectedPlayerId = null;
  clearRoot();
  addPitch(graph);
  graph.players.forEach((player) => {
    const object = makePlayer(player);
    playerObjects.set(player.id, object);
    root.add(object);
  });
  addBall(graph);
  if (graph.carrier_id != null) setSelectedPlayer(graph.carrier_id, false);
  else updateSelectionText();
  refreshPlayerMarkers();
  setCamera('tactical');
  setStatus('已加载站位场景：可拖动视角，点击球员可观察');
  stats.innerHTML = `
    <dt>schema</dt><dd>${graph.schema_version}</dd>
    <dt>players</dt><dd>${graph.players.length}</dd>
    <dt>suggested</dt><dd>${graph.carrier_id == null ? 'none' : `#${graph.carrier_id}`}</dd>
    <dt>pitch</dt><dd>${graph.pitch.method}</dd>
    <dt>mode</dt><dd>position-first</dd>
  `;
}

function getPlayerDirection(player) {
  const direction = player?.direction?.three ? v3(player.direction.three) : new THREE.Vector3(1, 0, 0);
  direction.y = 0;
  if (direction.lengthSq() < 0.0001) direction.set(1, 0, 0);
  return direction.normalize();
}

function buildPlayerCamera(player, mode) {
  const position = v3(player.engines.three.position);
  const direction = getPlayerDirection(player);
  const lookAt = position.clone().add(direction.clone().multiplyScalar(mode === 'fpv' ? 16 : 10));
  lookAt.y = mode === 'fpv' ? 1.55 : 1.25;

  if (mode === 'fpv') {
    return {
      fov: 88,
      position: position.clone().setY(1.65),
      lookAt,
      controls: false,
    };
  }

  return {
    fov: 64,
    position: position.clone().sub(direction.clone().multiplyScalar(8)).setY(5.2),
    lookAt,
    controls: true,
  };
}

function setCamera(mode) {
  if (!currentGraph) {
    setStatus('还没有加载 scene_graph.json', true);
    return;
  }
  currentCameraMode = mode;
  if (mode === 'fpv') {
    const player = getObservationPlayer();
    if (!player) {
      setStatus('没有可观察的球员', true);
      return;
    }
    if (selectedPlayerId == null) setSelectedPlayer(player.id, false);
    const cam = buildPlayerCamera(player, 'fpv');
    camera.fov = cam.fov;
    camera.position.copy(cam.position);
    camera.lookAt(cam.lookAt);
    controls.target.copy(cam.lookAt);
    controls.enabled = cam.controls;
    setStatus(`当前视角：#${player.id} 第一人称`);
  } else if (mode === 'follow') {
    const player = getObservationPlayer();
    if (!player) {
      setStatus('没有可观察的球员', true);
      return;
    }
    if (selectedPlayerId == null) setSelectedPlayer(player.id, false);
    const cam = buildPlayerCamera(player, 'follow');
    camera.fov = cam.fov;
    camera.position.copy(cam.position);
    camera.lookAt(cam.lookAt);
    controls.target.copy(cam.lookAt);
    controls.enabled = cam.controls;
    setStatus(`当前视角：#${player.id} 跟随`);
  } else {
    camera.fov = currentGraph.cameras.tactical?.fov_degrees || 58;
    camera.position.set(0, 78, 0.1);
    controls.target.set(0, 0, 0);
    camera.lookAt(controls.target);
    controls.enabled = true;
    setStatus('当前视角：战术俯视，可点击球员选择观察对象');
  }
  camera.updateProjectionMatrix();
  controls.update();
}

document.querySelectorAll('[data-camera]').forEach((button) => {
  button.addEventListener('click', () => setCamera(button.dataset.camera));
});
document.querySelector('#showSourceImage').addEventListener('click', showSourceImage);
document.querySelector('#closeSourceImage').addEventListener('click', hideSourceImage);
imageModal.addEventListener('click', (event) => {
  if (event.target === imageModal) hideSourceImage();
});
window.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') hideSourceImage();
});

let pointerDown = null;
renderer.domElement.addEventListener('pointerdown', (event) => {
  pointerDown = { x: event.clientX, y: event.clientY };
});

renderer.domElement.addEventListener('pointerup', (event) => {
  if (!currentGraph || !pointerDown) return;
  const moved = Math.hypot(event.clientX - pointerDown.x, event.clientY - pointerDown.y);
  pointerDown = null;
  if (moved > 5) return;

  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects([...playerObjects.values()], true);
  const hit = hits.find((item) => item.object.userData.playerId != null);
  if (!hit) return;
  setSelectedPlayer(hit.object.userData.playerId);
  setStatus(`已选中 #${hit.object.userData.playerId}，可切换跟随或第一人称`);
});

document.querySelector('#fileInput').addEventListener('change', async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    renderGraph(JSON.parse(await file.text()));
  } catch (error) {
    console.error(error);
    setStatus(`JSON 加载失败：${error.message}`, true);
  }
});

document.querySelector('#imageInput').addEventListener('change', async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('image', file);

  try {
    setStatus(`正在识别 ${file.name}，首次加载模型会稍慢...`);
    const response = await fetch('/api/process', {
      method: 'POST',
      body: formData,
    });
    const contentType = response.headers.get('content-type') || '';
    if (!contentType.includes('application/json')) {
      const text = await response.text();
      const hint = text.trim().startsWith('<!') || text.trim().startsWith('<html')
        ? '服务返回了 HTML。请确认是用 python3 server.py 启动，而不是 python3 -m http.server。'
        : text.slice(0, 180);
      throw new Error(hint);
    }
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    renderGraph(payload.scene_graph);
    setStatus(`已生成 scene graph：${payload.scene_graph.source?.image_path || file.name}`);
  } catch (error) {
    console.error(error);
    setStatus(`图片处理失败：${error.message}`, true);
  } finally {
    event.target.value = '';
  }
});

async function loadDefault() {
  try {
    const response = await fetch('../output_real/scene_graph.json', { cache: 'no-store' });
    if (response.ok) renderGraph(await response.json());
    else setStatus('未找到默认 output_real/scene_graph.json，可手动选择文件', true);
  } catch (error) {
    console.info('默认 scene_graph.json 未加载，可手动选择文件。', error);
    setStatus('默认 scene_graph.json 未加载，可手动选择文件', true);
  }
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
loadDefault();
animate();
