import React, { useEffect, useRef } from 'react';
import { Animated, StyleSheet, View, Text } from 'react-native';

interface CameraOverlayProps {
  faceDetected: boolean;
  livenessStatus: 'idle' | 'checking' | 'passed' | 'failed';
  instruction: string;
}

/**
 * CameraOverlay — transparent SVG-style reticle layered over the camera preview.
 * Guides the user to position their face inside the oval target zone.
 */
const CameraOverlay: React.FC<CameraOverlayProps> = ({
  faceDetected,
  livenessStatus,
  instruction,
}) => {
  const pulseAnim = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    if (faceDetected) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, {
            toValue: 1.04,
            duration: 600,
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnim, {
            toValue: 1,
            duration: 600,
            useNativeDriver: true,
          }),
        ]),
      ).start();
    } else {
      pulseAnim.stopAnimation();
      pulseAnim.setValue(1);
    }
  }, [faceDetected, pulseAnim]);

  const borderColor = (() => {
    if (livenessStatus === 'passed') return '#22c55e';
    if (livenessStatus === 'failed') return '#ef4444';
    if (faceDetected) return '#3b82f6';
    return '#475569';
  })();

  const statusIcon = (() => {
    if (livenessStatus === 'passed') return '✓';
    if (livenessStatus === 'failed') return '✗';
    if (faceDetected) return '';
    return '';
  })();

  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      {/* Dark vignette corners */}
      <View style={styles.vignette} />

      {/* Oval face target */}
      <View style={styles.ovalContainer}>
        <Animated.View
          style={[
            styles.oval,
            { borderColor, transform: [{ scale: pulseAnim }] },
          ]}
        />
        {statusIcon ? (
          <Text style={[styles.statusIcon, { color: borderColor }]}>
            {statusIcon}
          </Text>
        ) : null}
      </View>

      {/* Corner brackets */}
      <View style={styles.bracketTL} />
      <View style={[styles.bracketTL, styles.bracketTR]} />
      <View style={[styles.bracketTL, styles.bracketBL]} />
      <View style={[styles.bracketTL, styles.bracketBR]} />

      {/* Instruction — top during liveness so bottom capture bar never covers it */}
      <View
        style={[
          styles.instructionBox,
          livenessStatus === 'checking' ? styles.instructionTop : styles.instructionBottom,
        ]}>
        <Text style={styles.instructionText}>{instruction}</Text>
      </View>
    </View>
  );
};

const OVAL_W = 220;
const OVAL_H = 280;
const BRACKET = 28;
const BRACKET_THICK = 3;

const styles = StyleSheet.create({
  vignette: {
    position: 'absolute',
    top: 0, left: 0, right: 0, bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.35)',
  },
  ovalContainer: {
    position: 'absolute',
    top: '15%',
    alignSelf: 'center',
    width: OVAL_W,
    height: OVAL_H,
    alignItems: 'center',
    justifyContent: 'center',
  },
  oval: {
    width: OVAL_W,
    height: OVAL_H,
    borderRadius: OVAL_W / 2,
    borderWidth: 2.5,
    backgroundColor: 'transparent',
  },
  statusIcon: {
    position: 'absolute',
    fontSize: 40,
    fontWeight: '800',
  },
  bracketTL: {
    position: 'absolute',
    top: '13%',
    left: '10%',
    width: BRACKET,
    height: BRACKET,
    borderTopWidth: BRACKET_THICK,
    borderLeftWidth: BRACKET_THICK,
    borderColor: '#f8fafc',
    borderRadius: 4,
  },
  bracketTR: {
    left: undefined,
    right: '10%',
    borderLeftWidth: 0,
    borderRightWidth: BRACKET_THICK,
  },
  bracketBL: {
    top: undefined,
    bottom: '13%',
    borderTopWidth: 0,
    borderBottomWidth: BRACKET_THICK,
  },
  bracketBR: {
    top: undefined,
    left: undefined,
    right: '10%',
    bottom: '13%',
    borderTopWidth: 0,
    borderLeftWidth: 0,
    borderRightWidth: BRACKET_THICK,
    borderBottomWidth: BRACKET_THICK,
  },
  instructionBox: {
    position: 'absolute',
    left: 16,
    right: 16,
    alignSelf: 'center',
    backgroundColor: 'rgba(0,0,0,0.75)',
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 12,
    zIndex: 10,
    elevation: 10,
  },
  instructionTop: {
    top: 56,
  },
  instructionBottom: {
    bottom: 120,
  },
  instructionText: {
    color: '#f8fafc',
    fontSize: 15,
    fontWeight: '600',
    textAlign: 'center',
  },
});

export default CameraOverlay;
