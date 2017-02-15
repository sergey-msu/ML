using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using ML.Core;
using ML.Core.Mathematics;
using ML.Core.ComputingNetworks;

namespace ML.ConsoleTest
{
  class Program
  {
    static readonly RandomGenerator m_Generator = new RandomGenerator();

    static void Main(string[] args)
    {
      testNewNetworkArchitecture();

      //generateNormal2Classes(200, 200);
      //generateNormal3Classes(100, 100, 100);

      //var file = "primitive.csv";
      //var file = "iris.csv";
      //var file = "iris.trunk.2d.csv";
      //var file = "normal.3classes.100.csv";
      //var file = "normal.2classes.1000.csv";
      var file = "normal.2classes.200.csv";
      //var file = "normal.3classes.1000.csv";
      //var file = "primitive3.csv";
      //var file = "ionosphere.csv";
      //var file = "sonar.csv";
      //var file = "breast-cancer.csv";

      var data = new DataWrapper(file);
      var test = new TestWrapper(data);
      test.Run();

      Console.WriteLine("DONE");
      Console.ReadLine();
    }

    static void testNewNetworkArchitecture()
    {
      var network = new MultNetwork();

      var layer1 = new SummLayer();
      network.AddFirstLayer(layer1);

      var layer2 = new DoubleLayer();
      layer1.AddNext(layer2);

      var output = new ThresholdLayer(2);
      layer2.AddNext(output);

      var x = new Point2D(0.7, 0.4);
      var y = network.Calculate(x);
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
  }

  #region Test Canvas

  public class MultNetwork : ComputingNetwork<Point2D, int> {}

  public class SummLayer : HiddenLayer<Point2D, double>
  {
    protected override bool DoUpdateParams(double[] pars, bool isDelta, ref int cursor)
    {
      return true;
    }

    protected override double DoCalculate(Point2D input)
    {
      return input.X + input.Y;
    }

    public override void Compile()
    {
      throw new NotImplementedException();
    }

    public override bool TrySetParam(int idx, double value, bool isDelta)
    {
      throw new NotImplementedException();
    }

    protected override bool DoTryGetParam(int idx, out double value)
    {
      throw new NotImplementedException();
    }
  }

  public class DoubleLayer : HiddenLayer<double, double>
  {
    protected override bool DoUpdateParams(double[] pars, bool isDelta, ref int cursor)
    {
      return true;
    }

    protected override double DoCalculate(double input)
    {
      return input * 2;
    }

    public override void Compile()
    {
      throw new NotImplementedException();
    }

    public override bool TrySetParam(int idx, double value, bool isDelta)
    {
      throw new NotImplementedException();
    }

    protected override bool DoTryGetParam(int idx, out double value)
    {
      throw new NotImplementedException();
    }
  }

  public class ThresholdLayer : OutputLayer<double, int>
  {
    private double m_Threshold;

    public ThresholdLayer(double threshold)
    {
      m_Threshold = threshold;
    }

    public override int Calculate(double input)
    {
      return (input > m_Threshold) ? 1 : 0;
    }

    public override void Compile()
    {
      throw new NotImplementedException();
    }

    public override bool TryUpdateParams(double[] pars, bool isDelta, ref int cursor)
    {
      if (pars==null || pars.Length <= cursor) return false;
      var value = pars[cursor];
      if (isDelta)
        m_Threshold += value;
      else
        m_Threshold = value;

      cursor++;

      return true;
    }

    public override bool TrySetParam(int idx, double value, bool isDelta)
    {
      if (idx != 0) return false;
      if (isDelta)
        m_Threshold += value;
      else
        m_Threshold = value;

      return true;
    }

    public override bool TryGetParam(int idx, out double value)
    {
      value = 0;
      if (idx != 0) return false;
      value = m_Threshold;
      return true;
    }
  }

  #endregion
}
