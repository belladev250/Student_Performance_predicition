import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:hive/hive.dart';
import 'package:hive_flutter/hive_flutter.dart'; // Add to pubspec.yaml

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  _PredictionPageState createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  final _formKey = GlobalKey<FormState>();
  final apiUrl = "http://127.0.0.1:8000/predict";

  // Form fields (11 original inputs)
  int school = 0;
  int sex = 0;
  int age = 16;
  int Medu = 2;
  int Fedu = 2;
  int studytime = 2;
  int failures = 0;
  int famrel = 4;
  int absences = 4;
  int G1 = 10;
  int G2 = 10;

  // New state variables
  double? predictedGrade;
  List<String> interventions = [];
  double confidence = 0.8;
  String errorMessage = '';
  bool isLoading = false;
  List<Map<String, dynamic>> predictionHistory = [];

  @override
  void initState() {
    super.initState();
    _initHive();
  }

  Future<void> _initHive() async {
    await Hive.initFlutter();
    await Hive.openBox('predictions');
    _loadHistory();
  }

  void _loadHistory() {
    final history = Hive.box('predictions').values.toList();
    setState(
      () => predictionHistory = List<Map<String, dynamic>>.from(history),
    );
  }

  // In your predictPerformance() method, modify the API response handling:

  Future<void> predictPerformance() async {
    if (_formKey.currentState!.validate()) {
      setState(() {
        isLoading = true;
        errorMessage = '';
        interventions = []; // Clear previous interventions
      });

      try {
        // For testing purposes only - remove in production
        const useMockData = false;
        if (useMockData) {
          await Future.delayed(const Duration(seconds: 1));
          setState(() {
            predictedGrade = 16.5;
            interventions = [
              "Join math study group (Tuesdays 4-6pm)",
              "Schedule meeting with advisor",
              "Increase study time by 2 hours/week",
            ];
            confidence = 0.85;
          });
          _savePrediction();
          return;
        }

        final response = await http.post(
          Uri.parse(apiUrl),
          headers: {'Content-Type': 'application/json'},
          body: json.encode({
            'school': school,
            'sex': sex,
            'age': age,
            'Medu': Medu,
            'Fedu': Fedu,
            'studytime': studytime,
            'failures': failures,
            'famrel': famrel,
            'absences': absences,
            'G1': G1,
            'G2': G2,
          }),
        );

        if (response.statusCode == 200) {
          final data = json.decode(response.body);
          setState(() {
            predictedGrade = data['predicted_grade']?.toDouble();

            // Ensure interventions are always displayed - even if empty
            interventions = List<String>.from(
              data['interventions'] ??
                  [
                    "No specific recommendations available",
                    "Maintain current study habits",
                  ],
            );

            confidence = data['confidence']?.toDouble() ?? 0.8;
          });
          _savePrediction();
        } else {
          throw Exception('API returned status code ${response.statusCode}');
        }
      } catch (e) {
        setState(() {
          errorMessage = 'Error: ${e.toString()}';
          // Provide fallback recommendations
          interventions = [
            "Could not connect to prediction service",
            "Try again later or contact support",
          ];
        });
      } finally {
        setState(() => isLoading = false);
      }
    }
  }

  void _savePrediction() {
    final entry = {
      'date': DateTime.now().toIso8601String(),
      'prediction': predictedGrade,
      'G1': G1,
      'G2': G2,
      'interventions': interventions,
    };
    Hive.box('predictions').add(entry);
    _loadHistory();
  }

  Widget _buildSection(String title, List<Widget> children) {
    return Card(
      margin: const EdgeInsets.only(bottom: 16),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title, style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 12),
            ...children,
          ],
        ),
      ),
    );
  }

  Widget _buildDropdown(
    String label,
    int value,
    List<String> options,
    Function(int) onChanged,
  ) {
    return DropdownButtonFormField<int>(
      value: value,
      decoration: InputDecoration(labelText: label),
      items: options
          .asMap()
          .entries
          .map((e) => DropdownMenuItem(value: e.key, child: Text(e.value)))
          .toList(),
      onChanged: (v) => onChanged(v!),
    );
  }

  Widget _buildNumberInput(
    String label,
    int value,
    Function(String) onChanged, {
    int min = 0,
    int max = 100,
  }) {
    return TextFormField(
      initialValue: value.toString(),
      keyboardType: TextInputType.number,
      decoration: InputDecoration(labelText: label),
      validator: (v) =>
          v == null ||
              int.tryParse(v) == null ||
              int.parse(v) < min ||
              int.parse(v) > max
          ? 'Enter $min-$max'
          : null,
      onChanged: onChanged,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Student Performance Predictor'),
        elevation: 2,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Form(
          key: _formKey,
          child: Column(
            children: [
              // History Section
              if (predictionHistory.isNotEmpty)
                _buildSection('Previous Predictions', [
                  SizedBox(
                    height: 120,
                    child: ListView.builder(
                      scrollDirection: Axis.horizontal,
                      itemCount: predictionHistory.length,
                      itemBuilder: (ctx, i) => Card(
                        child: Padding(
                          padding: const EdgeInsets.all(12),
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Text(
                                predictionHistory[i]['date']
                                    .toString()
                                    .substring(0, 10),
                              ),
                              Text(
                                'G3: ${predictionHistory[i]['prediction'].toStringAsFixed(1)}',
                              ),
                              Text(
                                'G1/G2: ${predictionHistory[i]['G1']}/${predictionHistory[i]['G2']}',
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ),
                ]),

              // Input Sections
              _buildSection('Student Information', [
                _buildDropdown('School', school, [
                  'GP',
                  'MS',
                ], (v) => setState(() => school = v)),
                const SizedBox(height: 12),
                _buildDropdown('Gender', sex, [
                  'Female',
                  'Male',
                ], (v) => setState(() => sex = v)),
                const SizedBox(height: 12),
                _buildNumberInput(
                  'Age',
                  age,
                  (v) => setState(() => age = int.parse(v)),
                  min: 15,
                  max: 22,
                ),
              ]),

              _buildSection('Family Background', [
                _buildDropdown("Mother's Education", Medu, [
                  'None',
                  'Primary',
                  'Secondary',
                  'Higher',
                  'Graduate',
                ], (v) => setState(() => Medu = v)),
                const SizedBox(height: 12),
                _buildDropdown("Father's Education", Fedu, [
                  'None',
                  'Primary',
                  'Secondary',
                  'Higher',
                  'Graduate',
                ], (v) => setState(() => Fedu = v)),
              ]),

              _buildSection('Academic History', [
                _buildNumberInput(
                  'Weekly Study Time (1-4)',
                  studytime,
                  (v) => setState(() => studytime = int.parse(v)),
                  min: 1,
                  max: 4,
                ),
                const SizedBox(height: 12),
                _buildNumberInput(
                  'Past Failures',
                  failures,
                  (v) => setState(() => failures = int.parse(v)),
                  min: 0,
                  max: 4,
                ),
                const SizedBox(height: 12),
                _buildNumberInput(
                  'Family Relationship (1-5)',
                  famrel,
                  (v) => setState(() => famrel = int.parse(v)),
                  min: 1,
                  max: 5,
                ),
              ]),

              _buildSection('Grades', [
                _buildNumberInput(
                  'First Period Grade (G1)',
                  G1,
                  (v) => setState(() => G1 = int.parse(v)),
                  min: 0,
                  max: 20,
                ),
                const SizedBox(height: 12),
                _buildNumberInput(
                  'Second Period Grade (G2)',
                  G2,
                  (v) => setState(() => G2 = int.parse(v)),
                  min: 0,
                  max: 20,
                ),
              ]),

              // Prediction Button
              SizedBox(
                width: double.infinity,
                height: 50,
                child: ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                  ),
                  onPressed: isLoading ? null : predictPerformance,
                  child: isLoading
                      ? const CircularProgressIndicator(color: Colors.white)
                      : const Text(
                          'PREDICT FINAL GRADE',
                          style: TextStyle(fontSize: 16),
                        ),
                ),
              ),

              // Error/Results Display
              if (errorMessage.isNotEmpty)
                Padding(
                  padding: const EdgeInsets.only(top: 16),
                  child: Text(
                    errorMessage,
                    style: TextStyle(
                      color: Theme.of(context).colorScheme.error,
                    ),
                  ),
                ),

              if (predictedGrade != null)
                _buildSection('Prediction Results', [
                  Row(
                    children: [
                      Text(
                        'Predicted G3 Grade:',
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                      const Spacer(),
                      Text(
                        predictedGrade!.toStringAsFixed(1),
                        style: const TextStyle(
                          fontSize: 28,
                          fontWeight: FontWeight.bold,
                          color: Colors.blue,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  LinearProgressIndicator(
                    value: predictedGrade! / 20,
                    minHeight: 20,
                    color: predictedGrade! >= 10 ? Colors.green : Colors.orange,
                  ),
                  const SizedBox(height: 16),
                  Text(
                    'Model Confidence: ${(confidence * 100).toStringAsFixed(0)}%',
                  ),
                  LinearProgressIndicator(
                    value: confidence,
                    color: Colors.blue[400],
                    backgroundColor: Colors.grey[200],
                  ),
                  if (interventions.isNotEmpty) ...[
                    const SizedBox(height: 16),
                    const Text(
                      'Recommended Actions:',
                      style: TextStyle(fontWeight: FontWeight.bold),
                    ),
                    ...interventions
                        .map(
                          (i) => Padding(
                            padding: const EdgeInsets.symmetric(vertical: 6),
                            child: Row(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Icon(
                                  Icons.arrow_right,
                                  size: 20,
                                  color: Colors.green,
                                ),
                                const SizedBox(width: 8),
                                Expanded(child: Text(i)),
                              ],
                            ),
                          ),
                        )
                        .toList(),
                  ],
                ]),

              // Disclaimer
              const Padding(
                padding: EdgeInsets.only(top: 24),
                child: Text(
                  'Note: This tool provides estimates based on historical patterns. '
                  'Individual results may vary. Always consult with educators.',
                  style: TextStyle(color: Colors.grey, fontSize: 12),
                  textAlign: TextAlign.center,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
