import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:hallar/hallar.dart';

void main() {
  final fixture = '${Directory.current.path}/../fixtures/tiny.hallar';
  final docs = {
    '0': 'el perro y el gato',
    '1': 'cafe con leche caliente',
    '2': 'receta de croquetas de jamon',
    '3': 'boleto de tren para el viaje',
  };

  HallarIndex make(HallarModel m) => HallarIndex(m)..addAll(docs);

  test('model', () {
    final m = HallarModel.open(fixture);
    expect(m.dimension, 8);
    expect(m.vocabularySize, 32);
    expect(m.tokenize('Hola mundos!'), [2, 3, 4, 23]);
    final v = m.embed('hola mundo');
    expect(v.fold<double>(0, (a, x) => a + x * x), closeTo(1.0, 1e-5));
    expect(m.embed('zzz').every((x) => x == 0), isTrue);
    expect(hallarVersion, isNotEmpty);
    m.close();
    expect(() => m.dimension, throwsStateError);
  });

  test('errors', () {
    expect(() => HallarModel.open('/nope/x.hallar'),
        throwsA(isA<HallarException>().having((e) => e.message, 'message', contains('cannot open'))));
    final m = HallarModel.open(fixture);
    final idx = make(m);
    expect(() => idx.add('0', 'dup'), throwsA(isA<HallarException>()));
    expect(() => idx.setVectorWeight(2), throwsA(isA<HallarException>()));
    idx.close();
    idx.close();
    expect(() => idx.search('x'), throwsStateError);
  });

  test('search modes', () {
    final idx = make(HallarModel.open(fixture));
    expect(idx.length, 4);
    expect(idx.contains('2'), isTrue);
    expect(idx.contains('9'), isFalse);
    expect(idx.search('croquetas de jamón', limit: 1, mode: SearchMode.keyword).first.id, '2');
    expect(idx.search('boleto tren').first.id, '3');
    expect(idx.search('perro', limit: 2, mode: SearchMode.vector).length, 2);
    expect(idx.search('zzz', mode: SearchMode.keyword), isEmpty);
  });

  test('unicode round trip', () {
    final idx = HallarIndex(HallarModel.open(fixture));
    const text = 'café 🚀 con leche 👨‍👩‍👧 ñandú 数学';
    idx.add('emoji-🔥', text);
    final hit = idx.search('cafe leche', limit: 1, mode: SearchMode.keyword).single;
    expect(hit.id, 'emoji-🔥');
    expect(hit.text, text);
  });

  test('save and load', () {
    final m = HallarModel.open(fixture);
    final idx = make(m);
    final dir = Directory.systemTemp.createTempSync('hallar-dart');
    try {
      idx.save(dir.path);
      final back = HallarIndex.load(m, dir.path);
      expect(back.length, 4);
      for (final q in ['perro', 'cafe', 'viaje']) {
        expect(back.search(q), idx.search(q));
      }
    } finally {
      dir.deleteSync(recursive: true);
    }
  });
}
