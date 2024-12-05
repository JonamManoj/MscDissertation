import 'package:flutter/material.dart';
import 'pages/home_page.dart';

void main() {
  runApp(const CKDApp());
}

class CKDApp extends StatelessWidget {
  const CKDApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'CKD Analyser',
      theme: ThemeData(
        primarySwatch: Colors.teal,
        brightness: Brightness.light,
      ),
      home: const HomePage(),
    );
  }
}
