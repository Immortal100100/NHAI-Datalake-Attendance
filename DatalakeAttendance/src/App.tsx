import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { View, Text, StyleSheet } from 'react-native';

import RegisterScreen from './screens/RegisterScreen';
import CheckInScreen from './screens/CheckInScreen';
import AdminSyncScreen from './screens/AdminSyncScreen';
import SyncBootstrap from './components/SyncBootstrap';

export type RootTabParamList = {
  Register: undefined;
  CheckIn: undefined;
  AdminSync: undefined;
};

const Tab = createBottomTabNavigator<RootTabParamList>();

const TabDot: React.FC<{ focused: boolean }> = ({ focused }) =>
  focused ? <View style={styles.activeDot} /> : <View style={styles.inactiveDot} />;

const App: React.FC = () => (
  <NavigationContainer>
    <SyncBootstrap />
    <Tab.Navigator
      screenOptions={{
        headerStyle: { backgroundColor: '#0f172a', elevation: 0, shadowOpacity: 0 },
        headerTintColor: '#f8fafc',
        headerTitleStyle: { fontWeight: '700', fontSize: 17 },
        tabBarStyle: styles.tabBar,
        tabBarActiveTintColor: '#60a5fa',
        tabBarInactiveTintColor: '#475569',
        tabBarShowLabel: true,
        tabBarLabelStyle: styles.tabLabel,
      }}>
      <Tab.Screen
        name="Register"
        component={RegisterScreen}
        options={{
          title: 'Enrol Biometrics',
          tabBarLabel: 'Enrol',
          tabBarIcon: ({ focused }) => <TabDot focused={focused} />,
        }}
      />
      <Tab.Screen
        name="CheckIn"
        component={CheckInScreen}
        options={{
          title: 'Mark Attendance',
          tabBarLabel: 'Check In',
          tabBarIcon: ({ focused }) => <TabDot focused={focused} />,
        }}
      />
      <Tab.Screen
        name="AdminSync"
        component={AdminSyncScreen}
        options={{
          title: 'Sync Panel',
          tabBarLabel: 'Sync',
          tabBarIcon: ({ focused }) => <TabDot focused={focused} />,
        }}
      />
    </Tab.Navigator>
  </NavigationContainer>
);

const styles = StyleSheet.create({
  tabBar: {
    backgroundColor: '#0f172a',
    borderTopColor: '#1e293b',
    borderTopWidth: 1,
    height: 64,
    paddingBottom: 8,
    paddingTop: 6,
  },
  tabLabel: {
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 0.3,
  },
  activeDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#60a5fa',
    marginBottom: 2,
  },
  inactiveDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: 'transparent',
    marginBottom: 2,
  },
});

export default App;
