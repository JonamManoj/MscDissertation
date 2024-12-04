import 'package:flutter/material.dart';

class ResultPage extends StatelessWidget {
  final String diagnosis;
  final double confidenceCKD;
  final double confidenceNoCKD;

  const ResultPage({
    Key? key,
    required this.diagnosis,
    required this.confidenceCKD,
    required this.confidenceNoCKD,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final bool isPositive = diagnosis.contains("YES");

    return Scaffold(
      appBar: AppBar(
        title: const Text('Diagnosis Result'),
      ),
      body: Center(
        child: Card(
          elevation: 8.0,
          margin: const EdgeInsets.all(16.0),
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  diagnosis,
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                    color: isPositive ? Colors.red : Colors.green,
                  ),
                ),
                const SizedBox(height: 20),
                Text(
                  isPositive
                      ? 'Confidence CKD: $confidenceCKD%\nPlease consult a doctor for further diagnosis.'
                      : 'Confidence No CKD: $confidenceNoCKD%\nYou are healthy. Keep maintaining a good lifestyle!',
                  style: const TextStyle(fontSize: 16),
                  textAlign: TextAlign.center,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
