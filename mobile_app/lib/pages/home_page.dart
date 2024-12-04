import 'package:flutter/material.dart';
import 'form_page.dart';
import 'result_page.dart';

class HomePage extends StatefulWidget {
  const HomePage({Key? key}) : super(key: key);

  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  int _selectedIndex = 0;

  final List<Widget> _pages = [
    const FormPage(),
    const ResultPage(
      diagnosis: "No diagnosis yet",
      confidenceCKD: 0.0,
      confidenceNoCKD: 0.0,
    ),
  ];

  void _onTabSelected(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('CKD Predictor'),
      ),
      body: _pages[_selectedIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _selectedIndex,
        onTap: _onTabSelected,
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.assignment), label: 'Form'),
          BottomNavigationBarItem(icon: Icon(Icons.analytics), label: 'Result'),
        ],
      ),
    );
  }
}
