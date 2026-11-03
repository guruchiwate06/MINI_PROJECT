import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useSignalStore } from '../../store/signalStore';
import { Scene3DOverlays } from '../../types';

interface Class230TrainProps {
  overlays: Scene3DOverlays;
  onSelectSensor?: (channelId: number) => void;
}

/**
 * Detailed 3D Model of the Pop-Up Metro Class 230 (Vivarail D78)
 * Hybrid Passenger Railcar.
 *
 * Sourced from physical vehicle geometry:
 * Length: 18.3m, Width: 2.8m, Height: 3.65m above rail plane.
 * Standard Track Gauge: 1.435m.
 * Bogie Centers: 13.5m spacing (±6.75m from vehicle center).
 *
 * Features physically based materials (PBR), rotating flanged steel wheelsets,
 * primary/secondary suspension, underfloor hybrid power packs,
 * passenger cabin windows, and physical telemetry sensor mounts (Ch 1-6).
 */
export const Class230Train: React.FC<Class230TrainProps> = ({ overlays, onSelectSensor }) => {
  const trainRef = useRef<THREE.Group>(null);
  const wheelsRef = useRef<THREE.Group[]>([]);
  const { currentDistance, currentSpeed, spatialZone, activeChannel, currentRunMeta } = useSignalStore();

  // Materials with PBR calibration
  const materials = useMemo(() => {
    return {
      // Main body: Sleek dark metallic livery reminiscent of modern commuter rolling stock
      bodyCar: new THREE.MeshStandardMaterial({
        color: new THREE.Color('#161a22'),
        metalness: 0.85,
        roughness: 0.25,
        envMapIntensity: 1.2
      }),
      // Livery accent: Signature cyan transit racing stripe
      stripeCyan: new THREE.MeshStandardMaterial({
        color: new THREE.Color('#06b6d4'),
        emissive: new THREE.Color('#06b6d4'),
        emissiveIntensity: 0.35,
        metalness: 0.6,
        roughness: 0.3
      }),
      // Warning yellow front panel & door edges
      warningYellow: new THREE.MeshStandardMaterial({
        color: new THREE.Color('#f59e0b'),
        metalness: 0.3,
        roughness: 0.4
      }),
      // Cabin roof: matte anti-reflective coated aluminum
      roofMetal: new THREE.MeshStandardMaterial({
        color: new THREE.Color('#222834'),
        metalness: 0.7,
        roughness: 0.5
      }),
      // Tinted passenger windows
      windowGlass: new THREE.MeshPhysicalMaterial({
        color: new THREE.Color('#0b131e'),
        metalness: 0.1,
        roughness: 0.05,
        transmission: 0.65,
        thickness: 0.05,
        transparent: true,
        opacity: 0.88,
        reflectivity: 0.9
      }),
      // Cab windshield (slightly clearer)
      windshieldGlass: new THREE.MeshPhysicalMaterial({
        color: new THREE.Color('#102235'),
        metalness: 0.1,
        roughness: 0.03,
        transmission: 0.75,
        thickness: 0.08,
        transparent: true,
        opacity: 0.82
      }),
      // Bogie cast steel frame & heavy iron components
      bogieSteel: new THREE.MeshStandardMaterial({
        color: new THREE.Color('#1e2430'),
        metalness: 0.8,
        roughness: 0.65
      }),
      // Polished steel wheel running tread
      wheelTread: new THREE.MeshStandardMaterial({
        color: new THREE.Color('#d4d8e0'),
        metalness: 0.98,
        roughness: 0.12,
        envMapIntensity: 1.5
      }),
      // Wheel dark cast iron hub
      wheelHub: new THREE.MeshStandardMaterial({
        color: new THREE.Color('#262d3a'),
        metalness: 0.85,
        roughness: 0.45
      }),
      // Primary helical spring dampers
      springAmber: new THREE.MeshStandardMaterial({
        color: new THREE.Color('#d97706'),
        metalness: 0.8,
        roughness: 0.3
      }),
      // Secondary air suspension rubber bellows
      airBellows: new THREE.MeshStandardMaterial({
        color: new THREE.Color('#111419'),
        metalness: 0.1,
        roughness: 0.9
      }),
      // Undercarriage equipment boxes & battery enclosures
      undercarriageBox: new THREE.MeshStandardMaterial({
        color: new THREE.Color('#131720'),
        metalness: 0.7,
        roughness: 0.6
      }),
      // Headlights active LED emissive
      headlightLed: new THREE.MeshStandardMaterial({
        color: new THREE.Color('#ffffff'),
        emissive: new THREE.Color('#ecfeff'),
        emissiveIntensity: 4.5,
        toneMapped: false
      }),
      // Tail lights red marker LED
      tailLightLed: new THREE.MeshStandardMaterial({
        color: new THREE.Color('#ef4444'),
        emissive: new THREE.Color('#ef4444'),
        emissiveIntensity: 3.0,
        toneMapped: false
      }),
      // Sensor active pulse node
      sensorActive: new THREE.MeshStandardMaterial({
        color: new THREE.Color('#06b6d4'),
        emissive: new THREE.Color('#06b6d4'),
        emissiveIntensity: 3.5,
        toneMapped: false
      }),
      // Sensor default idle node
      sensorIdle: new THREE.MeshStandardMaterial({
        color: new THREE.Color('#0e7490'),
        emissive: new THREE.Color('#0e7490'),
        emissiveIntensity: 1.2,
        toneMapped: false
      })
    };
  }, []);

  // Animate train position along track and rotate wheels synchronously
  useFrame((_, delta) => {
    if (!trainRef.current) return;

    // Longitudinal position along Z axis follows recorded chainage in meters
    const targetZ = currentDistance;
    trainRef.current.position.z = targetZ;

    // Procedural chassis oscillation and bogie pitch based on track zone
    // NOTE: This is an ILLUSTRATIVE PROCEDURAL ANIMATION, NOT an empirical double integration of acceleration.
    if (spatialZone === 'ANOMALY') {
      const freq = 35; // illustrative track impact frequency (Hz)
      const amp = 0.012; // illustrative 1.2 cm vertical peak shock oscillation
      trainRef.current.position.y = Math.sin(Date.now() * 0.001 * freq) * amp;
      trainRef.current.rotation.x = Math.sin(Date.now() * 0.001 * freq * 0.7) * 0.003;
    } else if (spatialZone === 'BUFFER') {
      const amp = 0.004; // illustrative 4 mm secondary ringing oscillation
      trainRef.current.position.y = Math.sin(Date.now() * 0.001 * 12) * amp;
      trainRef.current.rotation.x = 0;
    } else {
      // Normal smooth track rolling bounce
      const speedNorm = Math.min(1.0, currentSpeed / 25);
      trainRef.current.position.y = Math.sin(Date.now() * 0.001 * 4) * 0.0015 * speedNorm;
      trainRef.current.rotation.x = 0;
    }

    // Wheel rotation based on linear displacement: dtheta = ds / r
    // Wheel radius r = 0.425m (illustrative D78 nominal 840mm wheelset)
    const wheelRadius = 0.425;
    const speedMs = (currentSpeed * 1000) / 3600;
    const dirSign = currentRunMeta?.direction === 'BtoA' ? -1 : 1;
    const rotSpeed = dirSign * (speedMs / wheelRadius) * delta;

    wheelsRef.current.forEach((wheelGroup) => {
      if (wheelGroup) {
        wheelGroup.rotation.x += rotSpeed;
      }
    });
  });

  // Bogie Assembly Builder
  const renderBogie = (pivotZ: number, isFront: boolean) => {
    const axleSpacing = 1.1; // ±1.1m from pivot -> 2.2m wheelbase
    const wheelY = 0.575; // Axle centerline height above base plane
    const railX = 0.7175; // Standard gauge 1.435m / 2

    return (
      <group position={[0, 0, pivotZ]} key={`bogie-${pivotZ}`}>
        {/* Cast steel H-Frame Bogie Chassis */}
        <mesh position={[0, 0.72, 0]} material={materials.bogieSteel} castShadow receiveShadow>
          <boxGeometry args={[2.2, 0.16, 2.9]} />
        </mesh>

        {/* Bogie Side Frames (Left & Right) */}
        {[-1.05, 1.05].map((x, idx) => (
          <group key={`side-frame-${idx}`} position={[x, 0.65, 0]}>
            <mesh material={materials.bogieSteel} castShadow receiveShadow>
              <boxGeometry args={[0.18, 0.28, 3.1]} />
            </mesh>

            {/* Primary Coil Springs over Axle Boxes */}
            {[-axleSpacing, axleSpacing].map((z, sIdx) => (
              <group position={[0, 0, z]} key={`spring-${sIdx}`}>
                {/* Axle Box Enclosure */}
                <mesh position={[0, -0.08, 0]} material={materials.bogieSteel} castShadow>
                  <boxGeometry args={[0.26, 0.22, 0.26]} />
                </mesh>
                {/* Twin Helical Suspension Springs */}
                {[-0.08, 0.08].map((dz, spIdx) => (
                  <mesh key={`coil-${spIdx}`} position={[0, 0.1, dz]} material={materials.springAmber} castShadow>
                    <cylinderGeometry args={[0.06, 0.06, 0.24, 12]} />
                  </mesh>
                ))}
              </group>
            ))}
          </group>
        ))}

        {/* Secondary Air Suspension Bellows (Left & Right) */}
        {[-0.85, 0.85].map((x, idx) => (
          <mesh key={`air-spring-${idx}`} position={[x, 0.88, 0]} material={materials.airBellows} castShadow>
            <cylinderGeometry args={[0.22, 0.24, 0.22, 16]} />
          </mesh>
        ))}

        {/* Center Bolster Pivot Pin */}
        <mesh position={[0, 0.86, 0]} material={materials.bogieSteel} castShadow>
          <cylinderGeometry args={[0.18, 0.18, 0.25, 20]} />
        </mesh>

        {/* 2 Axle Sets with Flanged Steel Wheels */}
        {[-axleSpacing, axleSpacing].map((zOffset, axleIdx) => (
          <group
            key={`axle-set-${axleIdx}`}
            position={[0, wheelY, zOffset]}
            ref={(el) => {
              const globalIdx = (isFront ? 0 : 2) + axleIdx;
              if (el) wheelsRef.current[globalIdx] = el;
            }}
          >
            {/* Solid Steel Axle Shaft */}
            <mesh rotation={[0, 0, Math.PI / 2]} material={materials.wheelHub} castShadow>
              <cylinderGeometry args={[0.08, 0.08, 1.8, 16]} />
            </mesh>

            {/* Disc Brake Rotor on Axle */}
            <mesh rotation={[0, 0, Math.PI / 2]} material={materials.bogieSteel} castShadow>
              <cylinderGeometry args={[0.28, 0.28, 0.06, 24]} />
            </mesh>

            {/* Left & Right Flanged Railway Wheels */}
            {[-railX, railX].map((xPos, sideIdx) => {
              const flangeSign = xPos > 0 ? 1 : -1;
              return (
                <group position={[xPos, 0, 0]} key={`wheel-${sideIdx}`}>
                  {/* Wheel Running Tread (contact with rail head) */}
                  <mesh rotation={[0, 0, Math.PI / 2]} material={materials.wheelTread} castShadow>
                    <cylinderGeometry args={[0.425, 0.425, 0.13, 32]} />
                  </mesh>
                  {/* Inner Wheel Safety Flange (prevent derailment) */}
                  <mesh
                    position={[-0.075 * flangeSign, 0, 0]}
                    rotation={[0, 0, Math.PI / 2]}
                    material={materials.wheelTread}
                    castShadow
                  >
                    <cylinderGeometry args={[0.46, 0.46, 0.035, 32]} />
                  </mesh>
                  {/* Central Hub & Bolt Ring */}
                  <mesh rotation={[0, 0, Math.PI / 2]} material={materials.wheelHub}>
                    <cylinderGeometry args={[0.22, 0.22, 0.14, 20]} />
                  </mesh>
                </group>
              );
            })}
          </group>
        ))}

        {/* Physical Sensor Indicators on Bogie */}
        {overlays.showSensors && (
          <>
            {/* CH 1 / CH 3: Bogie Primary Suspension Accelerometer Node */}
            <group
              position={[1.16, 0.76, 0]}
              onClick={(e) => {
                e.stopPropagation();
                onSelectSensor?.(isFront ? 1 : 3);
              }}
            >
              <mesh material={activeChannel === (isFront ? 1 : 3) ? materials.sensorActive : materials.sensorIdle}>
                <boxGeometry args={[0.12, 0.12, 0.12]} />
              </mesh>
              <pointLight
                color="#06b6d4"
                intensity={activeChannel === (isFront ? 1 : 3) ? 1.8 : 0.6}
                distance={1.5}
              />
            </group>

            {/* CH 2 / CH 4: Axle Box Unsprung Accelerometer Node (front axle) */}
            <group
              position={[1.18, 0.58, axleSpacing]}
              onClick={(e) => {
                e.stopPropagation();
                onSelectSensor?.(isFront ? 2 : 4);
              }}
            >
              <mesh material={activeChannel === (isFront ? 2 : 4) ? materials.sensorActive : materials.sensorIdle}>
                <sphereGeometry args={[0.07, 16, 16]} />
              </mesh>
              <pointLight
                color="#06b6d4"
                intensity={activeChannel === (isFront ? 2 : 4) ? 2.2 : 0.8}
                distance={1.5}
              />
            </group>
          </>
        )}
      </group>
    );
  };

  return (
    <group ref={trainRef}>
      {/* ========================================================================= */}
      {/* 1. PASSENGER CAR MAIN BODY (Length: 18.3m, Width: 2.8m, Height: 2.7m)     */}
      {/* ========================================================================= */}
      <group position={[0, 2.3, 0]}>
        {/* Main Aluminum Commuter Bodyshell */}
        <mesh material={materials.bodyCar} castShadow receiveShadow>
          <boxGeometry args={[2.76, 2.5, 17.8]} />
        </mesh>

        {/* Vivarail Cyan High-Speed Accent Striping (Waistline) */}
        {[-1.39, 1.39].map((x, idx) => (
          <mesh key={`stripe-${idx}`} position={[x, -0.15, 0]} material={materials.stripeCyan}>
            <boxGeometry args={[0.02, 0.14, 17.8]} />
          </mesh>
        ))}

        {/* Passenger Cabin Panoramic Tinted Windows (Left & Right) */}
        {[-1.39, 1.39].map((x, sideIdx) => (
          <group key={`windows-side-${sideIdx}`} position={[x, 0.35, 0]}>
            {[-6.2, -4.3, -2.4, 0.0, 2.4, 4.3, 6.2].map((z, winIdx) => (
              <mesh key={`win-${winIdx}`} position={[0, 0, z]} material={materials.windowGlass} castShadow={false}>
                <boxGeometry args={[0.04, 0.95, 1.4]} />
              </mesh>
            ))}
          </group>
        ))}

        {/* Passenger Double Sliding Plug Doors (2 sets per side) */}
        {[-1.39, 1.39].map((x, sideIdx) => (
          <group key={`doors-side-${sideIdx}`} position={[x, -0.1, 0]}>
            {[-3.35, 3.35].map((z, doorIdx) => (
              <group key={`door-${doorIdx}`} position={[0, 0, z]}>
                {/* Yellow Safety Door Threshold Edge */}
                <mesh material={materials.warningYellow}>
                  <boxGeometry args={[0.03, 1.9, 1.3]} />
                </mesh>
                {/* Center rubber parting seam */}
                <mesh position={[0.01, 0, 0]} material={materials.bodyCar}>
                  <boxGeometry args={[0.025, 1.86, 1.22]} />
                </mesh>
                {/* Door Glass Panes */}
                <mesh position={[0.02, 0.35, 0]} material={materials.windowGlass}>
                  <boxGeometry args={[0.03, 0.85, 0.95]} />
                </mesh>
              </group>
            ))}
          </group>
        ))}

        {/* Curved Roof Crown */}
        <mesh position={[0, 1.32, 0]} material={materials.roofMetal} castShadow>
          <cylinderGeometry args={[1.36, 1.38, 17.8, 24, 1, false, -Math.PI / 2, Math.PI]} />
        </mesh>
      </group>

      {/* ========================================================================= */}
      {/* 2. AERODYNAMIC DRIVING CABS (Front Cab +Z, Rear Cab -Z)                   */}
      {/* ========================================================================= */}
      {/* FRONT DRIVING CAB (+Z = 8.9m) */}
      <group position={[0, 2.3, 8.9]}>
        {/* Tapered Nose Geometry */}
        <mesh position={[0, -0.15, 0.35]} material={materials.bodyCar} castShadow receiveShadow>
          <boxGeometry args={[2.72, 2.2, 0.8]} />
        </mesh>

        {/* Yellow Visibility Warning Wrap on Front Bumper */}
        <mesh position={[0, -0.65, 0.76]} material={materials.warningYellow} castShadow>
          <boxGeometry args={[2.5, 0.9, 0.1]} />
        </mesh>

        {/* Swept Panoramic Front Windshield */}
        <mesh position={[0, 0.32, 0.65]} rotation={[-0.18, 0, 0]} material={materials.windshieldGlass}>
          <boxGeometry args={[2.3, 0.95, 0.1]} />
        </mesh>

        {/* High-Beam LED Headlights (Left, Right, Center Top) */}
        {[-0.85, 0.85].map((x, idx) => (
          <group key={`headlight-${idx}`} position={[x, -0.4, 0.82]}>
            <mesh material={materials.headlightLed} rotation={[Math.PI / 2, 0, 0]}>
              <cylinderGeometry args={[0.11, 0.11, 0.05, 16]} />
            </mesh>
            <spotLight
              color="#f8fafc"
              intensity={8.0}
              distance={45.0}
              angle={0.42}
              penumbra={0.6}
              position={[0, 0, 0.1]}
              target-position={[x * 0.5, -0.4, 35.0]}
              castShadow
            />
          </group>
        ))}

        {/* Top Center Headlamp */}
        <mesh position={[0, 0.95, 0.45]} rotation={[Math.PI / 2, 0, 0]} material={materials.headlightLed}>
          <cylinderGeometry args={[0.09, 0.09, 0.05, 16]} />
        </mesh>

        {/* Red Marker Tail Lights (when stationary or reversed) */}
        {[-1.0, 1.0].map((x, idx) => (
          <mesh key={`tail-${idx}`} position={[x, -0.65, 0.82]} rotation={[Math.PI / 2, 0, 0]} material={materials.tailLightLed}>
            <cylinderGeometry args={[0.06, 0.06, 0.04, 16]} />
          </mesh>
        ))}

        {/* Front Automatic Coupler (Scharfenberg mechanical / electrical coupler) */}
        <group position={[0, -1.35, 0.7]}>
          <mesh material={materials.bogieSteel} castShadow>
            <boxGeometry args={[0.3, 0.25, 0.7]} />
          </mesh>
          <mesh position={[0, 0, 0.4]} rotation={[Math.PI / 2, 0, 0]} material={materials.wheelTread}>
            <cylinderGeometry args={[0.16, 0.16, 0.12, 16]} />
          </mesh>
        </group>
      </group>

      {/* REAR DRIVING CAB (-Z = -8.9m) */}
      <group position={[0, 2.3, -8.9]} rotation={[0, Math.PI, 0]}>
        <mesh position={[0, -0.15, 0.35]} material={materials.bodyCar} castShadow receiveShadow>
          <boxGeometry args={[2.72, 2.2, 0.8]} />
        </mesh>
        <mesh position={[0, -0.65, 0.76]} material={materials.warningYellow} castShadow>
          <boxGeometry args={[2.5, 0.9, 0.1]} />
        </mesh>
        <mesh position={[0, 0.32, 0.65]} rotation={[-0.18, 0, 0]} material={materials.windshieldGlass}>
          <boxGeometry args={[2.3, 0.95, 0.1]} />
        </mesh>
        {/* Red Tail Marker Lights Active on Rear */}
        {[-0.85, 0.85].map((x, idx) => (
          <group key={`rear-marker-${idx}`} position={[x, -0.4, 0.82]}>
            <mesh material={materials.tailLightLed} rotation={[Math.PI / 2, 0, 0]}>
              <cylinderGeometry args={[0.11, 0.11, 0.05, 16]} />
            </mesh>
            <pointLight color="#ef4444" intensity={2.5} distance={6.0} />
          </group>
        ))}
      </group>

      {/* ========================================================================= */}
      {/* 3. ROOF EQUIPMENT (HVAC, Resistor Packs, Antenna, GNSS Satellite Dome)    */}
      {/* ========================================================================= */}
      <group position={[0, 3.72, 0]}>
        {/* Twin Roof HVAC Units */}
        {[-4.2, 4.2].map((z, idx) => (
          <group key={`hvac-${idx}`} position={[0, 0.18, z]}>
            <mesh material={materials.roofMetal} castShadow>
              <boxGeometry args={[2.1, 0.36, 2.4]} />
            </mesh>
            {/* Ventilation Exhaust Fan Cowlings */}
            {[-0.5, 0.5].map((x, fIdx) => (
              <mesh key={`fan-${fIdx}`} position={[x, 0.19, 0]} material={materials.bogieSteel}>
                <cylinderGeometry args={[0.38, 0.38, 0.06, 16]} />
              </mesh>
            ))}
          </group>
        ))}

        {/* Dynamic Braking Resistor Grid */}
        <mesh position={[0, 0.14, 0]} material={materials.bogieSteel} castShadow>
          <boxGeometry args={[1.8, 0.28, 3.2]} />
        </mesh>

        {/* Survey GNSS Receiver Antenna Puck (Centerline) */}
        <group position={[0, 0.36, 1.5]}>
          <mesh material={materials.stripeCyan}>
            <cylinderGeometry args={[0.14, 0.14, 0.08, 16]} />
          </mesh>
          <pointLight color="#06b6d4" intensity={0.8} distance={2.0} />
        </group>
      </group>

      {/* ========================================================================= */}
      {/* 4. UNDERCARRIAGE HYBRID PROPULSION & BATTERY TRACTION ENCLOSURES          */}
      {/* ========================================================================= */}
      <group position={[0, 0.65, 0]}>
        {/* Vivarail Modular Diesel-Battery Hybrid Power Pack Box */}
        <mesh position={[0, 0.15, -1.2]} material={materials.undercarriageBox} castShadow>
          <boxGeometry args={[2.3, 0.55, 3.8]} />
        </mesh>

        {/* High-Voltage Lithium-Ion Traction Battery Enclosure */}
        <mesh position={[0, 0.15, 1.8]} material={materials.undercarriageBox} castShadow>
          <boxGeometry args={[2.3, 0.55, 3.4]} />
        </mesh>

        {/* Auxiliary Compressed Air Reservoirs (Cylindrical Tanks) */}
        {[-0.8, 0.8].map((x, idx) => (
          <mesh key={`tank-${idx}`} position={[x, 0.05, -3.8]} rotation={[Math.PI / 2, 0, 0]} material={materials.bogieSteel} castShadow>
            <cylinderGeometry args={[0.24, 0.24, 1.6, 16]} />
          </mesh>
        ))}

        {/* High-Voltage Orange Shielded Cable Ducts */}
        <mesh position={[0.95, 0.38, 0]} material={materials.springAmber}>
          <boxGeometry args={[0.08, 0.08, 12.0]} />
        </mesh>
      </group>

      {/* ========================================================================= */}
      {/* 5. FRONT & REAR BOGIE ASSEMBLIES (±6.75m from vehicle center)             */}
      {/* ========================================================================= */}
      {renderBogie(6.75, true)}
      {renderBogie(-6.75, false)}

      {/* ========================================================================= */}
      {/* 6. CAR BODY SENSORS (CH 5: Front Floor, CH 6: Mid Body Floor)             */}
      {/* ========================================================================= */}
      {overlays.showSensors && (
        <group>
          {/* CH 5: Front Car Body Passenger Cabin Floor */}
          <group
            position={[0, 1.06, 6.75]}
            onClick={(e) => {
              e.stopPropagation();
              onSelectSensor?.(5);
            }}
          >
            <mesh material={activeChannel === 5 ? materials.sensorActive : materials.sensorIdle}>
              <cylinderGeometry args={[0.1, 0.1, 0.08, 16]} />
            </mesh>
            <pointLight color="#06b6d4" intensity={activeChannel === 5 ? 2.5 : 0.8} distance={2.0} />
          </group>

          {/* CH 6: Mid Car Body Floor Centerline */}
          <group
            position={[0, 1.06, 0.0]}
            onClick={(e) => {
              e.stopPropagation();
              onSelectSensor?.(6);
            }}
          >
            <mesh material={activeChannel === 6 ? materials.sensorActive : materials.sensorIdle}>
              <cylinderGeometry args={[0.1, 0.1, 0.08, 16]} />
            </mesh>
            <pointLight color="#06b6d4" intensity={activeChannel === 6 ? 2.5 : 0.8} distance={2.0} />
          </group>

          {/* Optical Track Inspection Camera Frustum Mount (1.52m above rail, 15° pitch) */}
          {overlays.showCameraFrustum && (
            <group position={[0, 1.52, 8.4]} rotation={[0.26, 0, 0]}>
              {/* Camera Housing Body */}
              <mesh material={materials.bogieSteel}>
                <boxGeometry args={[0.16, 0.14, 0.22]} />
              </mesh>
              {/* Camera Lens Element */}
              <mesh position={[0, 0, 0.12]} rotation={[Math.PI / 2, 0, 0]} material={materials.windshieldGlass}>
                <cylinderGeometry args={[0.05, 0.05, 0.04, 16]} />
              </mesh>
              {/* Pyramidal Inspection Field-of-View Frustum */}
              <mesh position={[0, 0, 1.2]} rotation={[Math.PI / 2, 0, 0]}>
                <coneGeometry args={[0.7, 2.4, 4, 1, true]} />
                <meshBasicMaterial color="#06b6d4" wireframe transparent opacity={0.3} />
              </mesh>
            </group>
          )}
        </group>
      )}

      {/* Wireframe Holographic Overlay for Bogie Kinematic Inspection */}
      {overlays.showBogieWireframe && (
        <group>
          <mesh position={[0, 0.72, 6.75]}>
            <boxGeometry args={[2.3, 0.6, 3.2]} />
            <meshBasicMaterial color="#06b6d4" wireframe />
          </mesh>
          <mesh position={[0, 0.72, -6.75]}>
            <boxGeometry args={[2.3, 0.6, 3.2]} />
            <meshBasicMaterial color="#06b6d4" wireframe />
          </mesh>
        </group>
      )}
    </group>
  );
};
