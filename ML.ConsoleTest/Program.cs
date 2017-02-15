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
      var network = new MultiNetwork();

      var layer1 = new SummLayer();
      network.AddFirstLayer(layer1);

      var layer11 = new SimpleCompositeLayer(3, -1);
      layer11.AddSubLayer(new TripleLayer());
      layer11.AddSubLayer(new IncrementLayer());
      layer1.AddNext(layer11);

      var layer2 = new DoubleLayer();
      layer11.AddNext(layer2);

      var output = new ThresholdLayer(1);
      layer2.AddNext(output);

      var x = new Point2D(0.1, 0.2);

      double val;
      int idx = 3;
      network.TryGetParam(ref idx, out val);

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

  public class MultiNetwork : ComputingNetwork<Point2D, double> {}

  public class SummLayer : HiddenLayer<Point2D, double>
  {
    protected override double DoCalculate(Point2D input)
    {
      return input.X + input.Y;
    }

    protected override bool DoUpdateParams(double[] pars, bool isDelta, ref int cursor)
    {
      return true;
    }

    protected override bool DoSetParam(ref int idx, double value, bool isDelta)
    {
      return false;
    }

    protected override bool DoGetParam(ref int idx, out double value)
    {
      value = 0;
      return false;
    }
  }

  public class SimpleCompositeLayer : CompositeLayer<double, double>
  {
    private double m_W1;
    private double m_W2;

    public SimpleCompositeLayer(double w1, double w2)
    {
      m_W1 = w1;
      m_W2 = w2;
    }

    protected override bool DoGetParam(ref int idx, out double value)
    {
      if (idx==0)
      {
        value = m_W1;
        return true;
      }

      if (idx==1)
      {
        value = m_W2;
        return true;
      }

      value = 0;
      idx -= 2;
      return false;
    }

    protected override bool DoSetParam(ref int idx, double value, bool isDelta)
    {
      if (idx==0)
      {
        if (isDelta) m_W1 += value;
        else m_W1 = value;
        return true;
      }

      if (idx==1)
      {
        if (isDelta) m_W2 += value;
        else m_W2 = value;
        return true;
      }

      idx -= 2;
      return false;
    }

    protected override bool DoUpdateParams(double[] pars, bool isDelta, ref int cursor)
    {
      if (cursor+2 >= pars.Length) return false;

      var val1 = pars[0];
      var val2 = pars[1];
      if (isDelta)
      {
        m_W1 += val1;
        m_W2 += val2;
      }
      else
      {
        m_W1 = val1;
        m_W2 = val2;
      }

      cursor += 2;
      return true;
    }

    protected override double MergeResults(double[] results)
    {
      if (results==null || results.Length != 2)
        throw new MLException("Incorrect composite output");

      return m_W1*results[0] + m_W2*results[1];
    }
  }

  public class DoubleLayer : HiddenLayer<double, double>
  {
    protected override double DoCalculate(double input)
    {
      return input * 2;
    }
    protected override bool DoUpdateParams(double[] pars, bool isDelta, ref int cursor)
    {
      return true;
    }

    protected override bool DoSetParam(ref int idx, double value, bool isDelta)
    {
      return false;
    }

    protected override bool DoGetParam(ref int idx, out double value)
    {
      value = 0;
      return false;
    }
  }

  public class TripleLayer : OutputLayer<double, double>
  {
    public override double Calculate(double input)
    {
      return input*3;
    }

    public override bool TryUpdateParams(double[] pars, bool isDelta, ref int cursor)
    {
      return true;
    }

    public override bool TrySetParam(ref int idx, double value, bool isDelta)
    {
      return false;
    }

    public override bool TryGetParam(ref int idx, out double value)
    {
      value = 0;
      return false;
    }
  }

  public class IncrementLayer : OutputLayer<double, double>
  {
    public override double Calculate(double input)
    {
      return input + 1;
    }

    public override bool TryUpdateParams(double[] pars, bool isDelta, ref int cursor)
    {
      return true;
    }

    public override bool TrySetParam(ref int idx, double value, bool isDelta)
    {
      return false;
    }

    public override bool TryGetParam(ref int idx, out double value)
    {
      value = 0;
      return false;
    }
  }

  public class IdentityOutputLayer : OutputLayer<double, double>
  {
    public override double Calculate(double input)
    {
      return input;
    }

    public override bool TryUpdateParams(double[] pars, bool isDelta, ref int cursor)
    {
      return true;
    }

    public override bool TrySetParam(ref int idx, double value, bool isDelta)
    {
      return false;
    }

    public override bool TryGetParam(ref int idx, out double value)
    {
      value = 0;
      return false;
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

    public override bool TrySetParam(ref int idx, double value, bool isDelta)
    {
      if (idx != 0)
      {
        idx--;
        return false;
      }

      if (isDelta)
        m_Threshold += value;
      else
        m_Threshold = value;

      return true;
    }

    public override bool TryGetParam(ref int idx, out double value)
    {
      value = 0;
      if (idx != 0)
      {
        idx--;
        return false;
      }

      value = m_Threshold;
      return true;
    }
  }

  #endregion
}
