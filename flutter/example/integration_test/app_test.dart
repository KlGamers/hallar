import 'package:flutter_test/flutter_test.dart';
import 'package:hallar/hallar.dart';
import 'package:hallar_example/main.dart';
import 'package:integration_test/integration_test.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('searches on device', (tester) async {
    final idx = await load();
    expect(idx.length, 4);
    expect(idx.search('croquetas jamón', limit: 1).first.id, '2');
    idx.add('emoji', 'café 🚀 con chocolate');
    expect(idx.search('chocolate', limit: 1, mode: SearchMode.keyword).first.text, 'café 🚀 con chocolate');
  });
}
