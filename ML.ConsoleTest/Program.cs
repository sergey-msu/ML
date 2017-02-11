using System;
using System.IO;
using System.Linq;
using System.Xml.Linq;
using System.Collections.Generic;
using ML.Core;
using ML.Core.Mathematics;
using ML.NeuralMethods;
using ML.NeuralMethods.Algorithms;

namespace ML.ConsoleTest
{
  class Program
  {
    static readonly RandomGenerator m_Generator = new RandomGenerator();

    static void Main(string[] args)
    {
      //neuralNetworkTest();
      neuralNetworkAlgorithmTest();

      //generateNormal2Classes(200, 200);
      //generateNormal3Classes(100, 100, 100);

      //var file = "primitive.csv";
      //var file = "iris.csv";
      //var file = "iris.trunk.2d.csv";
      //var file = "normal.3classes.100.csv";
      //var file = "normal.2classes.1000.csv";
      //var file = "normal.2classes.200.csv";
      //var file = "normal.3classes.1000.csv";
      //var file = "primitive3.csv";
      //var file = "ionosphere.csv";
      //var file = "sonar.csv";
      var file = "breast-cancer.csv";

      var data = new DataWrapper(file);
      var test = new TestWrapper(data);
      test.Run();

      Console.WriteLine("DONE");
      Console.ReadLine();
    }

    private static void generateNormal2Classes(int n1, int n2)
    {
      using (var file = File.Open("test2.normal2.csv", FileMode.CreateNew))
      using (var writer = new StreamWriter(file))
      {
        writer.WriteLine("f1,f2,_class,_training");

        var s1 = n1/30;
        for (int i = 0; i < n1; i++)
        {
          var p1 = m_Generator.GenerateNormalPoint(0, 0, 1);
          writer.WriteLine("{0},{1},{2},{3}", Math.Round(p1.X, 4), Math.Round(p1.Y, 4), "Green", i % s1 == 0 ? 1 : 0);
        }

        var s2 = n2/30;
        for (int i = 0; i < n2; i++)
        {
          var p2 = m_Generator.GenerateNormalPoint(2.5, 0, 0.5);
          writer.WriteLine("{0},{1},{2},{3}", Math.Round(p2.X, 4), Math.Round(p2.Y, 4), "Red", i % s2 == 0 ? 1 : 0);
        }
      }
    }

    private static void generateNormal3Classes(int n1, int n2, int n3)
    {
      using (var file = File.Open("test3.normal3.csv", FileMode.Create))
      using (var writer = new StreamWriter(file))
      {
        writer.WriteLine("f1,f2,_class,_value,_training");

        var s1 = n1/20;
        for (int i = 0; i < n1; i++)
        {
          var p1 = m_Generator.GenerateNormalPoint(0, 0, 1);
          writer.WriteLine("{0},{1},{2},{3},{4}", Math.Round(p1.X, 4), Math.Round(p1.Y, 4), "Green", 1, i % s1 == 0 ? 1 : 0);
        }

        var s2 = n2/20;
        for (int i = 0; i < n2; i++)
        {
          var p2 = m_Generator.GenerateNormalPoint(2.5, 0, 0.5);
          writer.WriteLine("{0},{1},{2},{3},{4}", Math.Round(p2.X, 4), Math.Round(p2.Y, 4), "Red", 2, i % s2 == 0 ? 1 : 0);
        }

        var s3 = n3/20;
        for (int i = 0; i < n3; i++)
        {
          var p3 = m_Generator.GenerateNormalPoint(1.7, 1.8, 0.5);
          writer.WriteLine("{0},{1},{2},{3},{4}", Math.Round(p3.X, 4), Math.Round(p3.Y, 4), "Blue", 3, i % s3 == 0 ? 1 : 0);
        }
      }
    }

    private static void neuralNetworkTest()
    {
      var network = new NeuralNetwork();
      network.ActivationFunction = Registry.ActivationFunctions.Identity;

      var l1 = network.CreateLayer();
      var n11 = l1.CreateNeuron();
      n11[0] = 0.5D;
      var n12 = l1.CreateNeuron();
      n12[0] = 0.1D;
      n12[1] = 0.3D;
      var n13 = l1.CreateNeuron();
      n13[0] = 0.2D;
      n13[1] = 0.4D;
      var n14 = l1.CreateNeuron();
      n14[1] = 0.5D;
      n14[2] = 0.7D;

      var l2 = network.CreateLayer();
      var n21 = l2.CreateNeuron();
      n21[0] = 0.1D;
      n21[1] = 0.2D;
      n21[2] = 0.3D;
      var n22 = l2.CreateNeuron();
      n22[2] = 0.1D;
      n22[3] = 0.2D;

      var l3 = network.CreateLayer();
      var n31 = l3.CreateNeuron();
      n31[0] = 0.5D;
      n31[1] = 0.1D;

      var input = new Point(1.0D, 1.0D, 1.0D);
      var result = network.Calculate(input);

      var correct = 0.185D; // correct value for testing

      var newWeights = new double[] { 1.0D, 1.0D, 1.0D, 1.0D, 1.0D, 1.0D, 1.0D, 1.0D, 1.0D, 1.0D, 1.0D, 1.0D, 1.0D, 1.0D };
      network.UpdateWeights(newWeights, false);
      result = network.Calculate(input);

      correct = 9.0D;
    }

    private static void neuralNetworkAlgorithmTest()
    {
      var class1 = new Class("green", 0, 1);
      var class2 = new Class("red", 1, 2);
      var sample = new ClassifiedSample
      {
        { new Point(1, 0), class1 },
        { new Point(0, 1), class1 },
        { new Point(0, 0), class2 },
        { new Point(1, 1), class2 }
      };

      var algorithm = new NeuralNetworkSimpleAlgorithm(sample)
      {
        EpochCount = 1,
        Margin = 5.0D,
        Step = 0.1D
      };
      var net = algorithm.Network;
      net.ActivationFunction = Registry.ActivationFunctions.Exp;

      var layer1 = net.CreateLayer();
      layer1.NormOutput = true;
      var n11 = layer1.CreateNeuron();
      n11[0] = 0.01D;
      n11[1] = 0.01D;
      var n12 = layer1.CreateNeuron();
      n12[0] = 0.01D;
      n12[1] = 0.01D;

      var output = net.CreateLayer();
      var n21 = output.CreateNeuron();
      n21[0] = 0.01D;
      n21[1] = 0.01D;
      var n22 = output.CreateNeuron();
      n22[0] = 0.01D;
      n22[1] = 0.01D;

      algorithm.Train();
    }
  }
}
