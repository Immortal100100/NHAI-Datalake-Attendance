// gesture-handler MUST be the very first import for React Navigation
import 'react-native-gesture-handler';
import React from 'react';
import { StatusBar } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import AppNavigator from './src/App';

const App: React.FC = () => (
  <SafeAreaProvider>
    <StatusBar barStyle="light-content" backgroundColor="#0f172a" />
    <AppNavigator />
  </SafeAreaProvider>
);

export default App;
