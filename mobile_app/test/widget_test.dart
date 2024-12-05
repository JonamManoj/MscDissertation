import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ckd_analyser/main.dart'; // Import your main.dart file where CKDApp is defined

void main() {
  testWidgets('CKD Analyser smoke test', (WidgetTester tester) async {
    // Build the CKDApp widget
    await tester.pumpWidget(CKDApp());

    // Verify the initial UI is built correctly
    expect(find.text('CKD Analyser'), findsOneWidget); // Checks that the title is displayed correctly

    // Find text input fields
    final ageField = find.byType(TextField).first;
    final bpField = find.byType(TextField).at(1);

    // Enter some text into the text fields
    await tester.enterText(ageField, '61');
    await tester.enterText(bpField, '70');

    // Tap the predict button
    final predictButton = find.widgetWithText(ElevatedButton, 'Predict CKD');
    await tester.tap(predictButton);

    // Rebuild the widget after the state has changed
    await tester.pump();

    // Since this is a smoke test, we are only checking for basic behavior
    expect(find.text('CKD Analyser'), findsOneWidget);
  });
}
