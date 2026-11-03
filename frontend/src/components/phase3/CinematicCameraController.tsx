import React, { useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { useSignalStore } from '../../store/signalStore';
import { CameraViewMode } from '../../types';

interface CinematicCameraControllerProps {
  viewMode: CameraViewMode;
  targetAnomalyZ?: number | null;
}

/**
 * Multi-Mode Cinematic Camera Controller
 *
 * Implements smooth lerp interpolation between 6 cinematic camera perspectives:
 * 1. side-follow: 45° tracking perspective gliding beside the train profile.
 * 2. front-track: Low-angle rail view looking at the advancing locomotive cab.
 * 3. rear-track: Chase camera following train roof toward the horizon.
 * 4. rail-macro: Wheel-rail interface close-up at rail height.
 * 5. bogie-inspect: Forensic inspection focused on Bogie 1 & Ch 1/2 sensors.
 * 6. free-orbit: Full OrbitControls with touch/mouse interaction.
 */
export const CinematicCameraController: React.FC<CinematicCameraControllerProps> = ({
  viewMode,
  targetAnomalyZ
}) => {
  const { camera } = useThree();
  const { currentDistance } = useSignalStore();
  const currentLookAt = useRef<THREE.Vector3>(new THREE.Vector3(0, 2, 0));

  useFrame((_, delta) => {
    const trainZ = currentDistance;
    const lerpSpeed = Math.min(1.0, delta * 4.5); // Smooth cinematic damping

    // If focusing on a selected anomaly site:
    if (targetAnomalyZ !== undefined && targetAnomalyZ !== null) {
      const targetPos = new THREE.Vector3(3.5, 2.0, targetAnomalyZ + 4.0);
      const targetAim = new THREE.Vector3(0.72, 0.3, targetAnomalyZ);

      camera.position.lerp(targetPos, lerpSpeed);
      currentLookAt.current.lerp(targetAim, lerpSpeed);
      camera.lookAt(currentLookAt.current);
      return;
    }

    if (viewMode === 'free-orbit') {
      // Free orbit handled by OrbitControls
      return;
    }

    let desiredPos = new THREE.Vector3();
    let desiredLookAt = new THREE.Vector3();

    switch (viewMode) {
      case 'side-follow':
        // High-end cinematic documentary side profile
        desiredPos.set(9.5, 3.2, trainZ - 2.5);
        desiredLookAt.set(0, 1.8, trainZ + 1.5);
        break;

      case 'front-track':
        // Front tracking angle looking down the rails at the advancing train cab
        desiredPos.set(0.0, 1.35, trainZ + 16.0);
        desiredLookAt.set(0, 2.0, trainZ);
        break;

      case 'rear-track':
        // Trailing chase camera looking forward over the train's roof
        desiredPos.set(0.0, 4.2, trainZ - 18.0);
        desiredLookAt.set(0, 2.1, trainZ + 12.0);
        break;

      case 'rail-macro':
        // Rail-level wheel-rail interface close-up (looking at front wheel flange)
        desiredPos.set(1.15, 0.32, trainZ + 7.8);
        desiredLookAt.set(0.72, 0.22, trainZ + 6.0);
        break;

      case 'bogie-inspect':
        // Forensic macro inspection focused on Bogie 1 & Ch 1/2 sensors
        desiredPos.set(2.4, 0.82, trainZ + 6.75);
        desiredLookAt.set(0.65, 0.68, trainZ + 6.75);
        break;
    }

    // Smoothly interpolate position and lookAt target
    camera.position.lerp(desiredPos, lerpSpeed);
    currentLookAt.current.lerp(desiredLookAt, lerpSpeed);
    camera.lookAt(currentLookAt.current);
  });

  if (viewMode === 'free-orbit') {
    return (
      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.06}
        maxDistance={70.0}
        minDistance={2.0}
        target={[0, 2.0, currentDistance]}
      />
    );
  }

  return null;
};

