import * as THREE from "./vendor/three-0.185.1.module.min.js";

(function () {
  "use strict";

  var root = document.getElementById("login-page");
  var canvas = document.getElementById("culinary-canvas");
  if (!root || !canvas) return;

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var finePointer = window.matchMedia("(hover: hover) and (pointer: fine)").matches;
  var renderer;
  var scene;
  var camera;
  var world;
  var sculpture;
  var architecturalRings;
  var flourDust;
  var animationFrame = 0;
  var paused = false;
  var destroyed = false;
  var pointer = { x: 0, y: 0, targetX: 0, targetY: 0 };
  var floatingObjects = [];
  var frameSamples = [];
  var lastFrameTime = 0;
  var lastStatsStamp = 0;
  var introContext = null;

  function physicalMaterial(options) {
    return new THREE.MeshPhysicalMaterial(Object.assign({
      roughness: 0.42,
      metalness: 0.02,
      clearcoat: 0.18,
      clearcoatRoughness: 0.32
    }, options));
  }

  function setMeshShadows(object, castShadow, receiveShadow) {
    object.traverse(function (child) {
      if (!child.isMesh) return;
      child.castShadow = castShadow;
      child.receiveShadow = receiveShadow;
    });
  }

  function createCroissant(materials) {
    var group = new THREE.Group();
    var path = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-1.28, 0.18, 0.05),
      new THREE.Vector3(-1.08, -0.38, 0.12),
      new THREE.Vector3(-0.58, -0.78, 0.08),
      new THREE.Vector3(0, -0.94, 0),
      new THREE.Vector3(0.62, -0.74, -0.07),
      new THREE.Vector3(1.08, -0.30, 0.04),
      new THREE.Vector3(1.3, 0.22, 0.14)
    ]);
    var bodyGeometry = new THREE.TubeGeometry(path, 104, 0.39, 24, false);
    var body = new THREE.Mesh(bodyGeometry, materials.pastry);
    group.add(body);

    var ribPositions = [0.14, 0.28, 0.42, 0.57, 0.71, 0.85];
    ribPositions.forEach(function (pathPosition, index) {
      var ribGeometry = new THREE.TorusGeometry(0.405, 0.025, 8, 28);
      var rib = new THREE.Mesh(ribGeometry, index % 2 ? materials.pastryEdge : materials.pastryDark);
      var point = path.getPointAt(pathPosition);
      var tangent = path.getTangentAt(pathPosition).normalize();
      rib.position.copy(point);
      rib.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), tangent);
      group.add(rib);
    });

    var endGeometry = new THREE.SphereGeometry(0.32, 24, 16);
    var leftEnd = new THREE.Mesh(endGeometry, materials.pastryEdge);
    leftEnd.scale.set(0.72, 1.08, 0.86);
    leftEnd.position.copy(path.getPointAt(0));
    group.add(leftEnd);

    var rightEnd = leftEnd.clone();
    rightEnd.position.copy(path.getPointAt(1));
    group.add(rightEnd);

    group.position.set(-1.22, -0.55, 0.55);
    group.rotation.set(0.08, -0.28, -0.02);
    group.scale.setScalar(1.08);
    setMeshShadows(group, true, true);
    return group;
  }

  function createPorcelainCup(materials) {
    var group = new THREE.Group();
    var profile = [
      new THREE.Vector2(0.48, -0.62),
      new THREE.Vector2(0.59, -0.56),
      new THREE.Vector2(0.68, 0.34),
      new THREE.Vector2(0.65, 0.53),
      new THREE.Vector2(0.55, 0.61)
    ];
    var cupBody = new THREE.Mesh(new THREE.LatheGeometry(profile, 56), materials.porcelain);
    group.add(cupBody);

    var rim = new THREE.Mesh(new THREE.TorusGeometry(0.61, 0.045, 12, 48), materials.porcelainEdge);
    rim.rotation.x = Math.PI / 2;
    rim.position.y = 0.61;
    group.add(rim);

    var coffee = new THREE.Mesh(new THREE.CylinderGeometry(0.54, 0.54, 0.038, 48), materials.coffee);
    coffee.position.y = 0.602;
    group.add(coffee);

    var crema = new THREE.Mesh(new THREE.TorusGeometry(0.18, 0.025, 8, 36), materials.crema);
    crema.rotation.x = Math.PI / 2;
    crema.position.set(-0.06, 0.626, 0.03);
    crema.scale.y = 0.66;
    group.add(crema);

    var handle = new THREE.Mesh(new THREE.TorusGeometry(0.33, 0.09, 12, 36, Math.PI * 1.62), materials.porcelain);
    handle.position.set(0.66, 0.03, 0);
    handle.rotation.z = -0.46;
    group.add(handle);

    group.position.set(0.77, -1.41, 1.15);
    group.rotation.set(-0.08, -0.42, 0);
    group.scale.setScalar(0.96);
    setMeshShadows(group, true, true);
    return group;
  }

  function createEntremet(materials) {
    var group = new THREE.Group();
    var cake = new THREE.Mesh(new THREE.CylinderGeometry(0.82, 0.78, 0.68, 64), materials.cake);
    cake.position.y = -0.02;
    group.add(cake);

    var glaze = new THREE.Mesh(new THREE.CylinderGeometry(0.835, 0.82, 0.19, 64), materials.lacquer);
    glaze.position.y = 0.405;
    group.add(glaze);

    var glazeCrown = new THREE.Mesh(new THREE.SphereGeometry(0.76, 48, 24), materials.lacquer);
    glazeCrown.scale.set(1.08, 0.15, 1.08);
    glazeCrown.position.y = 0.5;
    group.add(glazeCrown);

    var creamCenter = new THREE.Mesh(new THREE.SphereGeometry(0.16, 24, 16), materials.cream);
    creamCenter.scale.y = 0.55;
    creamCenter.position.set(-0.18, 0.64, 0.04);
    group.add(creamCenter);

    for (var petalIndex = 0; petalIndex < 5; petalIndex += 1) {
      var petal = new THREE.Mesh(new THREE.SphereGeometry(0.18, 20, 12), materials.cream);
      var petalAngle = petalIndex / 5 * Math.PI * 2;
      petal.scale.set(1.2, 0.38, 0.72);
      petal.position.set(-0.18 + Math.cos(petalAngle) * 0.18, 0.64, 0.04 + Math.sin(petalAngle) * 0.18);
      petal.rotation.y = -petalAngle;
      group.add(petal);
    }

    var berry = new THREE.Mesh(new THREE.SphereGeometry(0.115, 24, 16), materials.cherry);
    berry.position.set(0.2, 0.72, -0.02);
    group.add(berry);

    var plate = new THREE.Mesh(new THREE.CylinderGeometry(1.04, 1.14, 0.08, 64), materials.graphiteMetal);
    plate.position.y = -0.43;
    group.add(plate);

    group.position.set(1.32, -1.17, -0.42);
    group.rotation.y = -0.24;
    group.scale.setScalar(0.92);
    setMeshShadows(group, true, true);
    return group;
  }

  function createCacaoPod(materials) {
    var group = new THREE.Group();
    var pod = new THREE.Mesh(new THREE.SphereGeometry(0.58, 32, 24), materials.cacao);
    pod.scale.set(0.58, 1.35, 0.52);
    group.add(pod);

    for (var ribIndex = 0; ribIndex < 5; ribIndex += 1) {
      var rib = new THREE.Mesh(new THREE.TorusGeometry(0.32 + ribIndex * 0.035, 0.018, 6, 32), materials.cacaoEdge);
      rib.rotation.x = Math.PI / 2;
      rib.position.y = -0.45 + ribIndex * 0.23;
      rib.scale.set(1, 0.7, 1);
      group.add(rib);
    }

    var stem = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.075, 0.38, 12), materials.stem);
    stem.position.y = 0.92;
    stem.rotation.z = -0.18;
    group.add(stem);

    group.position.set(-2.62, 0.72, -0.28);
    group.rotation.set(0.22, -0.2, -0.68);
    group.scale.setScalar(0.78);
    setMeshShadows(group, true, false);
    return group;
  }

  function createCoffeeCherryCluster(materials) {
    var group = new THREE.Group();
    var cherryGeometry = new THREE.SphereGeometry(0.22, 24, 16);
    var positions = [
      [0, 0, 0], [0.29, 0.1, 0.04], [0.13, 0.31, -0.05], [-0.22, 0.24, 0.02]
    ];
    positions.forEach(function (position, index) {
      var cherry = new THREE.Mesh(cherryGeometry, index === 2 ? materials.cherryBright : materials.cherry);
      cherry.position.set(position[0], position[1], position[2]);
      group.add(cherry);
    });

    var stem = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.035, 0.75, 10), materials.stem);
    stem.position.set(0.05, 0.57, 0);
    stem.rotation.z = -0.32;
    group.add(stem);

    var leaf = new THREE.Mesh(new THREE.SphereGeometry(0.34, 24, 12), materials.leaf);
    leaf.scale.set(1.15, 0.18, 0.48);
    leaf.position.set(0.38, 0.67, -0.02);
    leaf.rotation.set(0.12, 0.28, -0.32);
    group.add(leaf);

    group.position.set(2.04, 0.52, 0.35);
    group.rotation.set(-0.1, -0.3, 0.2);
    setMeshShadows(group, true, false);
    return group;
  }

  function createWheat(materials) {
    var group = new THREE.Group();
    var stem = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.025, 2.6, 8), materials.wheat);
    group.add(stem);

    var grainGeometry = new THREE.SphereGeometry(0.12, 14, 10);
    for (var grainIndex = 0; grainIndex < 8; grainIndex += 1) {
      var direction = grainIndex % 2 ? 1 : -1;
      var grain = new THREE.Mesh(grainGeometry, materials.wheat);
      grain.scale.set(0.58, 1.1, 0.5);
      grain.position.set(direction * 0.12, 0.28 + grainIndex * 0.19, 0);
      grain.rotation.z = direction * 0.56;
      group.add(grain);
    }

    group.position.set(-3.1, -0.6, -0.62);
    group.rotation.set(0.16, -0.2, -0.2);
    group.scale.setScalar(0.76);
    setMeshShadows(group, true, false);
    return group;
  }

  function createMacaron(materials) {
    var group = new THREE.Group();
    var shellGeometry = new THREE.CylinderGeometry(0.4, 0.4, 0.18, 40);
    var top = new THREE.Mesh(shellGeometry, materials.macaron);
    top.position.y = 0.14;
    group.add(top);
    var bottom = top.clone();
    bottom.position.y = -0.14;
    group.add(bottom);
    var filling = new THREE.Mesh(new THREE.CylinderGeometry(0.38, 0.38, 0.11, 40), materials.cream);
    group.add(filling);
    group.position.set(2.55, -1.46, 0.8);
    group.rotation.set(0.22, 0.48, -0.12);
    group.scale.setScalar(0.78);
    setMeshShadows(group, true, true);
    return group;
  }

  function createChocolateShard(materials) {
    var shard = new THREE.Mesh(new THREE.OctahedronGeometry(0.62, 1), materials.chocolate);
    shard.scale.set(0.58, 1.24, 0.2);
    shard.position.set(-2.25, -1.3, 1.16);
    shard.rotation.set(0.32, 0.18, -0.38);
    shard.castShadow = true;
    return shard;
  }

  function createSteam(materials) {
    var steamGroup = new THREE.Group();
    for (var steamIndex = 0; steamIndex < 3; steamIndex += 1) {
      var xOffset = (steamIndex - 1) * 0.17;
      var curve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(xOffset, 0, 0),
        new THREE.Vector3(xOffset - 0.13, 0.33, 0.02),
        new THREE.Vector3(xOffset + 0.09, 0.68, -0.02),
        new THREE.Vector3(xOffset - 0.04, 1.03, 0.01)
      ]);
      var steam = new THREE.Mesh(new THREE.TubeGeometry(curve, 30, 0.012, 6, false), materials.steam);
      steam.material = materials.steam.clone();
      steam.material.opacity = 0.25 + steamIndex * 0.1;
      steamGroup.add(steam);
    }
    steamGroup.position.set(0.72, -0.55, 1.15);
    return steamGroup;
  }

  function createFlourTexture() {
    var sprite = document.createElement("canvas");
    sprite.width = 32;
    sprite.height = 32;
    var context = sprite.getContext("2d");
    var gradient = context.createRadialGradient(16, 16, 0, 16, 16, 16);
    gradient.addColorStop(0, "rgba(235,232,222,0.92)");
    gradient.addColorStop(0.35, "rgba(235,232,222,0.42)");
    gradient.addColorStop(1, "rgba(235,232,222,0)");
    context.fillStyle = gradient;
    context.fillRect(0, 0, 32, 32);
    return new THREE.CanvasTexture(sprite);
  }

  function createFlourDust() {
    var count = window.innerWidth <= 820 ? 180 : 420;
    var positions = new Float32Array(count * 3);
    var scales = new Float32Array(count);
    for (var index = 0; index < count; index += 1) {
      positions[index * 3] = (Math.random() - 0.5) * 14;
      positions[index * 3 + 1] = (Math.random() - 0.5) * 8;
      positions[index * 3 + 2] = (Math.random() - 0.5) * 7 - 0.5;
      scales[index] = 0.4 + Math.random() * 0.8;
    }
    var geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("scale", new THREE.BufferAttribute(scales, 1));
    var material = new THREE.PointsMaterial({
      color: 0xd9d7cf,
      size: window.innerWidth <= 820 ? 0.035 : 0.045,
      map: createFlourTexture(),
      transparent: true,
      opacity: 0.38,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true
    });
    return new THREE.Points(geometry, material);
  }

  function createArchitecturalRings(materials) {
    var group = new THREE.Group();
    var ringData = [
      { radius: 3.45, tube: 0.018, x: -0.2, y: 0.15, z: -1.4, scaleY: 0.82, material: materials.chrome },
      { radius: 4.1, tube: 0.012, x: -0.45, y: 0.1, z: -1.8, scaleY: 0.72, material: materials.chromeMuted },
      { radius: 2.78, tube: 0.026, x: -0.05, y: 0.18, z: -1.1, scaleY: 0.88, material: materials.lacquerDark }
    ];
    ringData.forEach(function (data) {
      var ring = new THREE.Mesh(new THREE.TorusGeometry(data.radius, data.tube, 8, 160), data.material);
      ring.position.set(data.x, data.y, data.z);
      ring.scale.y = data.scaleY;
      group.add(ring);
    });
    return group;
  }

  function createPlatform(materials) {
    var group = new THREE.Group();
    var base = new THREE.Mesh(new THREE.CylinderGeometry(3.65, 3.9, 0.24, 96), materials.graphiteMetal);
    base.position.y = -2.16;
    base.receiveShadow = true;
    group.add(base);

    var bevel = new THREE.Mesh(new THREE.TorusGeometry(3.67, 0.08, 12, 120), materials.chromeMuted);
    bevel.rotation.x = Math.PI / 2;
    bevel.position.y = -2.04;
    group.add(bevel);

    var inset = new THREE.Mesh(new THREE.CylinderGeometry(3.1, 3.25, 0.035, 96), materials.platformInset);
    inset.position.y = -2.01;
    inset.receiveShadow = true;
    group.add(inset);
    return group;
  }

  function createSceneMaterials() {
    return {
      pastry: physicalMaterial({ color: 0xc9773f, roughness: 0.34, clearcoat: 0.34 }),
      pastryEdge: physicalMaterial({ color: 0xe6ad6f, roughness: 0.4, clearcoat: 0.28 }),
      pastryDark: physicalMaterial({ color: 0x7f3c2e, roughness: 0.52, clearcoat: 0.12 }),
      porcelain: physicalMaterial({ color: 0xd9d8d1, roughness: 0.22, clearcoat: 0.52 }),
      porcelainEdge: physicalMaterial({ color: 0xeff0eb, roughness: 0.18, clearcoat: 0.64 }),
      coffee: physicalMaterial({ color: 0x22100d, roughness: 0.18, clearcoat: 0.72 }),
      crema: physicalMaterial({ color: 0xba7952, roughness: 0.32, clearcoat: 0.48 }),
      cake: physicalMaterial({ color: 0xd0b8a1, roughness: 0.48, clearcoat: 0.16 }),
      cream: physicalMaterial({ color: 0xebe7dd, roughness: 0.38, clearcoat: 0.24 }),
      lacquer: physicalMaterial({ color: 0xa92d3d, roughness: 0.2, clearcoat: 0.86, clearcoatRoughness: 0.12 }),
      lacquerDark: physicalMaterial({ color: 0x6d1727, roughness: 0.25, clearcoat: 0.54, emissive: 0x210307, emissiveIntensity: 0.18 }),
      cherry: physicalMaterial({ color: 0x8f2133, roughness: 0.25, clearcoat: 0.72 }),
      cherryBright: physicalMaterial({ color: 0xc23d4e, roughness: 0.23, clearcoat: 0.8 }),
      cacao: physicalMaterial({ color: 0x6e3b29, roughness: 0.5, clearcoat: 0.16 }),
      cacaoEdge: physicalMaterial({ color: 0x9e6040, roughness: 0.48 }),
      stem: physicalMaterial({ color: 0x3f5749, roughness: 0.62 }),
      leaf: physicalMaterial({ color: 0x405c4c, roughness: 0.5, clearcoat: 0.22 }),
      wheat: physicalMaterial({ color: 0xb18a55, roughness: 0.58 }),
      macaron: physicalMaterial({ color: 0x9d5960, roughness: 0.44, clearcoat: 0.18 }),
      chocolate: physicalMaterial({ color: 0x241415, roughness: 0.38, clearcoat: 0.26 }),
      graphiteMetal: physicalMaterial({ color: 0x15191a, metalness: 0.72, roughness: 0.27, clearcoat: 0.36 }),
      platformInset: physicalMaterial({ color: 0x242728, metalness: 0.24, roughness: 0.42 }),
      chrome: physicalMaterial({ color: 0xaeb7b5, metalness: 0.86, roughness: 0.2, transparent: true, opacity: 0.42 }),
      chromeMuted: physicalMaterial({ color: 0x697270, metalness: 0.72, roughness: 0.32, transparent: true, opacity: 0.3 }),
      steam: new THREE.MeshBasicMaterial({ color: 0xd8dedb, transparent: true, opacity: 0.36, depthWrite: false })
    };
  }

  function configureLights() {
    scene.add(new THREE.HemisphereLight(0xdbe2de, 0x180d11, 1.22));

    var keyLight = new THREE.SpotLight(0xffd8bd, 68, 34, 0.52, 0.5, 1.45);
    keyLight.position.set(-4.8, 7.2, 7.4);
    keyLight.target.position.set(-1.0, -0.4, 0);
    keyLight.castShadow = window.innerWidth > 820;
    keyLight.shadow.mapSize.set(1024, 1024);
    keyLight.shadow.bias = -0.0004;
    scene.add(keyLight, keyLight.target);

    var coolFill = new THREE.PointLight(0x9cc9c5, 24, 22, 1.7);
    coolFill.position.set(5.2, 2.8, 5.2);
    scene.add(coolFill);

    var lacquerRim = new THREE.PointLight(0xd13249, 34, 20, 1.8);
    lacquerRim.position.set(-5.4, 0.3, 1.1);
    scene.add(lacquerRim);

    var lowGlow = new THREE.PointLight(0xc58860, 18, 16, 2.0);
    lowGlow.position.set(0, -3.6, 3.6);
    scene.add(lowGlow);
  }

  function configureScene() {
    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x0c0e0f, 0.046);

    camera = new THREE.PerspectiveCamera(34, 1, 0.1, 80);
    camera.position.set(0, 0.12, 13.4);
    camera.lookAt(0, -0.25, 0);

    var materials = createSceneMaterials();
    world = new THREE.Group();
    scene.add(world);

    architecturalRings = createArchitecturalRings(materials);
    world.add(architecturalRings);

    sculpture = new THREE.Group();
    sculpture.add(createPlatform(materials));
    sculpture.add(createCroissant(materials));
    sculpture.add(createPorcelainCup(materials));
    sculpture.add(createEntremet(materials));

    var cacaoPod = createCacaoPod(materials);
    var cherryCluster = createCoffeeCherryCluster(materials);
    var wheat = createWheat(materials);
    var macaron = createMacaron(materials);
    var chocolateShard = createChocolateShard(materials);
    var steam = createSteam(materials);
    sculpture.add(cacaoPod, cherryCluster, wheat, macaron, chocolateShard, steam);
    floatingObjects.push(
      { object: cacaoPod, baseY: cacaoPod.position.y, amplitude: 0.08, speed: 0.62, phase: 0.4 },
      { object: cherryCluster, baseY: cherryCluster.position.y, amplitude: 0.11, speed: 0.7, phase: 1.5 },
      { object: wheat, baseY: wheat.position.y, amplitude: 0.05, speed: 0.5, phase: 2.2 },
      { object: macaron, baseY: macaron.position.y, amplitude: 0.07, speed: 0.58, phase: 3.0 },
      { object: chocolateShard, baseY: chocolateShard.position.y, amplitude: 0.09, speed: 0.66, phase: 4.0 },
      { object: steam, baseY: steam.position.y, amplitude: 0.06, speed: 0.44, phase: 1.1 }
    );
    world.add(sculpture);

    flourDust = createFlourDust();
    scene.add(flourDust);

    configureLights();
  }

  function resizeScene() {
    if (!renderer || !camera || !world) return;
    var width = Math.max(1, root.clientWidth);
    var height = Math.max(1, root.clientHeight);
    var compact = width <= 820;
    var shortViewport = height <= 680;
    var dprCap = compact ? 1.25 : 1.75;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, dprCap));
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.fov = compact ? 42 : 34;
    camera.position.z = compact ? 14.8 : (shortViewport ? 14.2 : 13.4);
    camera.updateProjectionMatrix();

    if (compact) {
      world.position.set(0, 1.05, -0.35);
      world.scale.setScalar(width < 480 ? 0.67 : 0.76);
    } else {
      world.position.set(width < 1100 ? -1.65 : -2.15, shortViewport ? -0.12 : -0.02, 0);
      world.scale.setScalar(shortViewport ? 0.84 : 0.96);
    }
    renderer.render(scene, camera);
  }

  function recordFrame(time) {
    if (!lastFrameTime) {
      lastFrameTime = time;
      return;
    }
    var delta = time - lastFrameTime;
    lastFrameTime = time;
    if (delta > 0 && delta < 120) {
      frameSamples.push(delta);
      if (frameSamples.length > 90) frameSamples.shift();
    }
  }

  function renderFrame(time) {
    if (destroyed || paused) return;
    var elapsed = (time || Date.now()) * 0.001;
    recordFrame(time || performance.now());
    if (time - lastStatsStamp > 1200 && frameSamples.length) {
      var sampledTotal = frameSamples.reduce(function (sum, value) { return sum + value; }, 0);
      var sampledAverage = sampledTotal / frameSamples.length;
      root.dataset.cinematicFps = String(Math.round(1000 / sampledAverage));
      root.dataset.cinematicTriangles = String(renderer.info.render.triangles);
      lastStatsStamp = time;
    }

    pointer.x += (pointer.targetX - pointer.x) * 0.045;
    pointer.y += (pointer.targetY - pointer.y) * 0.045;
    world.rotation.y = pointer.x * 0.095;
    world.rotation.x = pointer.y * 0.045;
    sculpture.position.y = Math.sin(elapsed * 0.52) * 0.045;
    sculpture.rotation.y = Math.sin(elapsed * 0.22) * 0.028 + pointer.x * 0.04;
    architecturalRings.rotation.z = elapsed * 0.018;
    architecturalRings.rotation.y = Math.sin(elapsed * 0.18) * 0.025;
    flourDust.rotation.y = elapsed * 0.012;
    flourDust.position.y = Math.sin(elapsed * 0.18) * 0.08;

    floatingObjects.forEach(function (entry) {
      entry.object.position.y = entry.baseY + Math.sin(elapsed * entry.speed + entry.phase) * entry.amplitude;
      entry.object.rotation.y += 0.0007;
    });

    renderer.render(scene, camera);
    animationFrame = window.requestAnimationFrame(renderFrame);
  }

  function onPointerMove(event) {
    if (!finePointer || reduceMotion || root.classList.contains("hidden")) return;
    var bounds = root.getBoundingClientRect();
    pointer.targetX = ((event.clientX - bounds.left) / bounds.width - 0.5) * 2;
    pointer.targetY = ((event.clientY - bounds.top) / bounds.height - 0.5) * 2;
  }

  function onPointerLeave() {
    pointer.targetX = 0;
    pointer.targetY = 0;
  }

  function pauseScene() {
    if (paused) return;
    paused = true;
    root.dataset.cinematicMotion = "paused";
    window.cancelAnimationFrame(animationFrame);
    animationFrame = 0;
  }

  function playScene() {
    if (!renderer || reduceMotion || destroyed || !paused || root.classList.contains("hidden") || document.hidden) return;
    paused = false;
    root.dataset.cinematicMotion = "active";
    animationFrame = window.requestAnimationFrame(renderFrame);
  }

  function disposeScene() {
    destroyed = true;
    pauseScene();
    root.removeEventListener("pointermove", onPointerMove);
    root.removeEventListener("pointerleave", onPointerLeave);
    window.removeEventListener("resize", resizeScene);
    if (introContext) introContext.revert();
    if (scene) {
      scene.traverse(function (child) {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          var materials = Array.isArray(child.material) ? child.material : [child.material];
          materials.forEach(function (material) {
            if (material.map) material.map.dispose();
            material.dispose();
          });
        }
      });
    }
    if (renderer) renderer.dispose();
  }

  function runIntro() {
    if (!window.gsap || reduceMotion) return;
    introContext = window.gsap.context(function () {
      var timeline = window.gsap.timeline({ defaults: { ease: "power4.out" } });
      timeline
        .fromTo(".cinematic-header", { autoAlpha: 0, y: -12 }, { autoAlpha: 1, y: 0, duration: 0.7, clearProps: "opacity,visibility,transform" }, 0.08)
        .fromTo(".login-story-kicker", { autoAlpha: 0, x: -20 }, { autoAlpha: 1, x: 0, duration: 0.62, clearProps: "opacity,visibility,transform" }, 0.18)
        .fromTo(".login-story h2 span", { autoAlpha: 0, y: 44, rotate: 1.4 }, { autoAlpha: 1, y: 0, rotate: 0, duration: 0.9, stagger: 0.08, clearProps: "opacity,visibility,transform" }, 0.22)
        .fromTo(".login-story-copy, .login-module-rail", { autoAlpha: 0, y: 14 }, { autoAlpha: 1, y: 0, duration: 0.68, stagger: 0.08, clearProps: "opacity,visibility,transform" }, 0.48)
        .fromTo(".login-console", { autoAlpha: 0, x: 42, scale: 0.975 }, { autoAlpha: 1, x: 0, scale: 1, duration: 0.96, clearProps: "opacity,visibility,transform" }, 0.18)
        .fromTo(".console-heading, .login-field-label, .login-input-shell, #signin-btn, .console-footer", { autoAlpha: 0, y: 12 }, { autoAlpha: 1, y: 0, duration: 0.46, stagger: 0.045, clearProps: "opacity,visibility,transform" }, 0.48);
    }, root);
  }

  function getRuntimeStats() {
    var total = frameSamples.reduce(function (sum, value) { return sum + value; }, 0);
    var averageFrameMs = frameSamples.length ? total / frameSamples.length : null;
    return {
      ready: root.classList.contains("webgl-ready"),
      reducedMotion: reduceMotion,
      paused: paused,
      averageFrameMs: averageFrameMs,
      estimatedFps: averageFrameMs ? Math.round(1000 / averageFrameMs) : null,
      samples: frameSamples.length,
      renderCalls: renderer ? renderer.info.render.calls : null,
      triangles: renderer ? renderer.info.render.triangles : null,
      pixelRatio: renderer ? renderer.getPixelRatio() : null
    };
  }

  try {
    renderer = new THREE.WebGLRenderer({
      canvas: canvas,
      antialias: window.innerWidth > 820,
      alpha: true,
      powerPreference: "high-performance"
    });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.08;
    renderer.shadowMap.enabled = window.innerWidth > 820;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    renderer.setClearColor(0x0c0e0f, 0);

    configureScene();
    resizeScene();
    root.classList.add("webgl-ready");
    root.dataset.cinematicEngine = "threejs";
    root.dataset.cinematicMotion = reduceMotion ? "static" : "active";
    runIntro();

    if (finePointer && !reduceMotion) {
      root.addEventListener("pointermove", onPointerMove, { passive: true });
      root.addEventListener("pointerleave", onPointerLeave, { passive: true });
    }
    window.addEventListener("resize", resizeScene, { passive: true });

    if (reduceMotion) {
      renderer.render(scene, camera);
    } else {
      animationFrame = window.requestAnimationFrame(renderFrame);
    }

    new MutationObserver(function () {
      var shouldPause = root.classList.contains("hidden") || document.hidden;
      if (shouldPause) pauseScene();
      else playScene();
    }).observe(root, { attributes: true, attributeFilter: ["class"] });

    document.addEventListener("visibilitychange", function () {
      if (document.hidden || root.classList.contains("hidden")) pauseScene();
      else playScene();
    });

    window.BakeryCinematic3D = {
      getStats: getRuntimeStats,
      pause: pauseScene,
      play: playScene,
      resize: resizeScene,
      destroy: disposeScene
    };
  } catch (error) {
    root.classList.add("webgl-fallback");
    root.dataset.cinematicEngine = "fallback";
    root.dataset.cinematicMotion = "static";
    runIntro();
    window.BakeryCinematic3D = {
      getStats: function () {
        return { ready: false, reducedMotion: reduceMotion, error: String(error && error.message || error) };
      }
    };
    console.error("Bakery cinematic scene failed to initialize", error);
  }
}());
