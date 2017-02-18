using System;
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ML.Core.ComputingNetworks;
using ML.Core;

namespace ML.Tests
{
  [TestClass]
  public class ComputingNetworkTests
  {
    #region Inner

    public class ScalarProductOutputLayer : OutputLayer<double[], double>
    {
      private double[] m_Coeffs;

      public ScalarProductOutputLayer(params double[] coeffs)
      {
        m_Coeffs = coeffs;
      }

      public double[] Coeffs { get { return m_Coeffs; } }

      public override int ParamsCount { get { return m_Coeffs.Length; } }

      public override double Calculate(double[] input)
      {
        var result = 0.0D;
        for (int i=0; i< m_Coeffs.Length; i++)
          result += m_Coeffs[i]*input[i];

        return result;
      }

      protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor)
      {
        for (int i=0; i<m_Coeffs.Length; i++)
        {
          if (cursor+i >= pars.Length) break;
          if (isDelta)
            m_Coeffs[i] += pars[cursor+i];
          else
            m_Coeffs[i] = pars[cursor+i];
        }
      }

      protected override double DoGetParam(int idx)
      {
        return m_Coeffs[idx];
      }

      protected override void DoSetParam(int idx, double value, bool isDelta)
      {
        if (isDelta)
          m_Coeffs[idx] += value;
        else
          m_Coeffs[idx] = value;
      }
    }

    public class MatrixHiddenLayer : HiddenLayer<double[], double[]>
    {
      public double[] m_Coeffs;

      public MatrixHiddenLayer(double a11, double a12, double a21, double a22)
      {
        m_Coeffs = new[] { a11, a12, a21, a22 };
      }

      public double[] Coeffs { get { return m_Coeffs; } }

      public override int ParamsCount { get { return m_Coeffs.Length; } }

      protected override double[] DoCalculate(double[] input)
      {
        var y1 = m_Coeffs[0]*input[0] + m_Coeffs[1]*input[1];
        var y2 = m_Coeffs[1]*input[0] + m_Coeffs[3]*input[1];

        return new[] { y1, y2 };
      }

      protected override double DoGetParam(int idx)
      {
        return m_Coeffs[idx];
      }

      protected override void DoSetParam(int idx, double value, bool isDelta)
      {
        if (isDelta)
          m_Coeffs[idx] += value;
        else
          m_Coeffs[idx] = value;
      }

      protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor)
      {
        for (int i=0; i<m_Coeffs.Length; i++)
        {
          if (cursor+i >= pars.Length) break;
          if (isDelta)
            m_Coeffs[i] += pars[cursor+i];
          else
            m_Coeffs[i] = pars[cursor+i];
        }
      }
    }

    public class DoublingLayer : OutputLayer<double, double>
    {
      public override int ParamsCount { get { return 0; } }

      public override double Calculate(double input)
      {
        return input * 2;
      }

      protected override double DoGetParam(int idx) { return 0; }
      protected override void DoSetParam(int idx, double value, bool isDelta) { }
      protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor) { }
    }

    public class DoubleStepLayer : HiddenLayer<double, double>
    {
      public override int ParamsCount { get { return 0; } }

      protected override double DoCalculate(double input)
      {
        return 2 * (int)(input / 2);
      }

      protected override double DoGetParam(int idx) { return 0; }
      protected override void DoSetParam(int idx, double value, bool isDelta) { }
      protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor) { }
    }

    public class IdentityLayer : OutputLayer<double, double>
    {
      public override int ParamsCount { get { return 0; } }

      public override double Calculate(double input)
      {
        return input;
      }

      protected override double DoGetParam(int idx) { return 0; }
      protected override void DoSetParam(int idx, double value, bool isDelta) { }
      protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor) { }
    }

    public class MaxLayer : HiddenLayer<Point2D, double>
    {
      public override int ParamsCount { get { return 0; } }

      protected override double DoCalculate(Point2D input)
      {
        return Math.Max(input.X, input.Y);
      }

      protected override double DoGetParam(int idx) { return 0; }
      protected override void DoSetParam(int idx, double value, bool isDelta) { }
      protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor) { }
    }

    public class ShiftingLayer : OutputLayer<double, double>
    {
      public double m_Shift;

      public ShiftingLayer(double shift)
      {
        m_Shift = shift;
      }

      public override int ParamsCount { get { return 1; } }

      public double Shift { get { return m_Shift; } }

      public override double Calculate(double input)
      {
        return input + m_Shift;
      }

      protected override double DoGetParam(int idx)
      {
        return m_Shift;
      }

      protected override void DoSetParam(int idx, double value, bool isDelta)
      {
        if (isDelta)
          m_Shift += value;
        else
          m_Shift = value;
      }

      protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor)
      {
        if (isDelta)
          m_Shift += pars[cursor];
        else
          m_Shift = pars[cursor];
      }
    }

    public class PowerLayer : OutputLayer<double, double>
    {
      public double m_Power;

      public PowerLayer(double power)
      {
        m_Power = power;
      }

      public double Power { get { return m_Power; } }

      public override int ParamsCount { get { return 1; } }

      public override double Calculate(double input)
      {
        return Math.Pow(Math.Abs(input), m_Power);
      }

      protected override double DoGetParam(int idx)
      {
        return m_Power;
      }

      protected override void DoSetParam(int idx, double value, bool isDelta)
      {
        if (isDelta)
          m_Power += value;
        else
          m_Power = value;
      }

      protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor)
      {
        if (isDelta)
          m_Power += pars[cursor];
        else
          m_Power = pars[cursor];
      }
    }

    public class TestCompositeLayer : CompositeLayer<double, double>
    {
      public double[] m_Coeffs;

      public TestCompositeLayer(double w1, double w2)
      {
        m_Coeffs = new[] { w1, w2 };
      }

      public double[] Coeffs { get { return m_Coeffs; } }

      public override int ParamsCount { get { return m_Coeffs.Length; } }

      protected override double DoGetParam(int idx)
      {
        return m_Coeffs[idx];
      }

      protected override void DoSetParam(int idx, double value, bool isDelta)
      {
        if (isDelta)
          m_Coeffs[idx] += value;
        else
          m_Coeffs[idx] = value;
      }

      protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor)
      {
        for (int i=0; i<m_Coeffs.Length; i++)
        {
          if (cursor+i >= pars.Length) break;
          if (isDelta)
            m_Coeffs[i] += pars[cursor+i];
          else
            m_Coeffs[i] = pars[cursor+i];
        }
      }

      protected override double MergeResults(double[] results)
      {
        return m_Coeffs[0]*results[0] + m_Coeffs[1]*results[1];
      }
    }

    public class NMatrixLayer : HiddenLayer<double[], double[]>
    {
      private double[,] m_Coeffs;

      public NMatrixLayer(double[,] coeffs)
      {
        m_Coeffs = new double[coeffs.GetLength(0), coeffs.GetLength(1)];
        Array.Copy(coeffs, m_Coeffs, coeffs.Length);
      }

      public override int ParamsCount { get { return m_Coeffs.Length; } }

      protected override double[] DoCalculate(double[] input)
      {
        var dim = m_Coeffs.GetLength(0);
        var result = new double[dim];

        for (int i=0; i<dim; i++)
        {
          var sum = 0.0D;
          for (int j=0; j<input.Length; j++)
            sum += m_Coeffs[i,j]*input[j];
          result[i] = sum;
        }

        return result;
      }

      protected override double DoGetParam(int idx)
      {
        var i = idx / m_Coeffs.GetLength(0);
        var j = idx % m_Coeffs.GetLength(0);
        return m_Coeffs[i,j];
      }

      protected override void DoSetParam(int idx, double value, bool isDelta)
      {
        var i = idx / m_Coeffs.GetLength(0);
        var j = idx % m_Coeffs.GetLength(0);
        if (isDelta)
          m_Coeffs[i,j] += value;
        else
          m_Coeffs[i,j] = value;
      }

      protected override void DoUpdateParams(double[] pars, bool isDelta, int cursor)
      {
        for (int l=0; l<m_Coeffs.Length; l++)
        {
          if (cursor+l >= pars.Length) break;
          var i = l / m_Coeffs.GetLength(0);
          var j = l % m_Coeffs.GetLength(0);
          if (isDelta)
            m_Coeffs[i,j] += pars[cursor+l];
          else
            m_Coeffs[i,j] = pars[cursor+l];
        }
      }
    }

    #endregion

    #region OutputLayer

    [TestMethod]
    public void OutputLayer_Build()
    {
      var layer = new ScalarProductOutputLayer(2, -1);
      layer.Build();

      Assert.AreEqual(2, layer.ParamsCount);
    }

    [TestMethod]
    public void OutputLayer_Calculate()
    {
      var layer = new ScalarProductOutputLayer(2, -1);
      layer.Build();

      var input = new double[] { 3, 4 };
      var res = layer.Calculate(input);

      Assert.AreEqual(2, res);
    }

    [TestMethod]
    public void OutputLayer_TryGetParam()
    {
      var layer = new ScalarProductOutputLayer(2, -1);
      layer.Build();

      double par1;
      double par2;
      double par3;
      var res1 = layer.TryGetParam(0, out par1);
      var res2 = layer.TryGetParam(1, out par2);
      var res3 = layer.TryGetParam(2, out par3);

      Assert.IsTrue(res1);
      Assert.AreEqual(2,  par1);
      Assert.IsTrue(res2);
      Assert.AreEqual(-1, par2);
      Assert.IsFalse(res3);
      Assert.AreEqual(0, par3);
    }

    [TestMethod]
    public void OutputLayer_TrySetParam()
    {
      var layer = new ScalarProductOutputLayer(2, -1);
      layer.Build();

      var res1 = layer.TrySetParam(0, 3, false);
      var res2 = layer.TrySetParam(1, -2, true);
      var res3 = layer.TrySetParam(2, 5, true);

      Assert.IsTrue(res1);
      Assert.IsTrue(res2);
      Assert.IsFalse(res3);

      Assert.AreEqual(3,  layer.Coeffs[0]);
      Assert.AreEqual(-3, layer.Coeffs[1]);
    }

    [TestMethod]
    public void OutputLayer_TryUpdateParams()
    {
      var layer = new ScalarProductOutputLayer(2, -1);
      layer.Build();
      var pars = new double[] { 1, 2, 3, -4, -1, 3 };
      var cursor = 1;

      var res = layer.TryUpdateParams(pars, false, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(3, cursor);
      Assert.AreEqual(2, layer.Coeffs[0]);
      Assert.AreEqual(3, layer.Coeffs[1]);

      res = layer.TryUpdateParams(pars, true, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(5, cursor);
      Assert.AreEqual(-2, layer.Coeffs[0]);
      Assert.AreEqual(2, layer.Coeffs[1]);

      res = layer.TryUpdateParams(pars, false, ref cursor);
      Assert.IsFalse(res);
      Assert.AreEqual(5, cursor);
      Assert.AreEqual(-2, layer.Coeffs[0]);
      Assert.AreEqual(2, layer.Coeffs[1]);
    }

    #endregion

    #region HiddenLayer

    [TestMethod]
    [ExpectedException(typeof(MLException))]
    public void HiddenLayer_Build_NoNextLayer()
    {
      var layer = new MatrixHiddenLayer(1, 1, 1, -1);
      layer.Build();
    }

    [TestMethod]
    public void HiddenLayer_Build()
    {
      var output = new ScalarProductOutputLayer(2, -3);
      var layer = new MatrixHiddenLayer(1, 1, 1, -1);
      layer.AddNextLayer(output);
      layer.Build();

      Assert.AreEqual(4, layer.ParamsCount);
      Assert.AreEqual(2, layer.NextLayer.ParamsCount);
    }

    [TestMethod]
    public void HiddenLayer_Calculate()
    {
      var output = new ScalarProductOutputLayer(2, -3);
      var layer = new MatrixHiddenLayer(1, 1, 1, -1);
      layer.AddNextLayer(output);
      layer.Build();

      var input = new double[] { 3, 4 };
      var res = (double)layer.Calculate(input);

      Assert.AreEqual(17, res);
    }

    [TestMethod]
    public void HiddenLayer_TryGetParam()
    {
      var output = new ScalarProductOutputLayer(2, -3);
      var layer = new MatrixHiddenLayer(1, 1, 1, -1);
      layer.AddNextLayer(output);
      layer.Build();

      double a11;
      double a12;
      double a21;
      double a22;
      double w1;
      double w2;
      double w3;
      var res11 = layer.TryGetParam(0, out a11);
      var res12 = layer.TryGetParam(1, out a12);
      var res21 = layer.TryGetParam(2, out a21);
      var res22 = layer.TryGetParam(3, out a22);
      var res1 = layer.TryGetParam(4,  out w1);
      var res2 = layer.TryGetParam(5,  out w2);
      var res3 = layer.TryGetParam(6,  out w3);

      Assert.IsTrue(res11);
      Assert.AreEqual(1, a11);
      Assert.IsTrue(res12);
      Assert.AreEqual(1, a12);
      Assert.IsTrue(res21);
      Assert.AreEqual(1, a21);
      Assert.IsTrue(res22);
      Assert.AreEqual(-1, a22);
      Assert.IsTrue(res1);
      Assert.AreEqual(2, w1);
      Assert.IsTrue(res2);
      Assert.AreEqual(-3, w2);
      Assert.IsFalse(res3);
      Assert.AreEqual(0, w3);
    }

    [TestMethod]
    public void HiddenLayer_TrySetParam()
    {
      var output = new ScalarProductOutputLayer(2, -3);
      var layer = new MatrixHiddenLayer(1, 1, 1, -1);
      layer.AddNextLayer(output);
      layer.Build();

      var res11 = layer.TrySetParam(0,  1, false);
      var res12 = layer.TrySetParam(1,  1, true);
      var res21 = layer.TrySetParam(2, -1, false);
      var res22 = layer.TrySetParam(3, -1, true);
      var res1  = layer.TrySetParam(4,  3, false);
      var res2  = layer.TrySetParam(5,  4, true);
      var res3  = layer.TrySetParam(6,  5, false);

      Assert.IsTrue(res11);
      Assert.AreEqual(1, layer.Coeffs[0]);
      Assert.IsTrue(res12);
      Assert.AreEqual(2, layer.Coeffs[1]);
      Assert.IsTrue(res21);
      Assert.AreEqual(-1, layer.Coeffs[2]);
      Assert.IsTrue(res22);
      Assert.AreEqual(-2, layer.Coeffs[3]);
      Assert.IsTrue(res1);
      Assert.AreEqual(3, output.Coeffs[0]);
      Assert.IsTrue(res2);
      Assert.AreEqual(1, output.Coeffs[1]);
      Assert.IsFalse(res3);
    }

    [TestMethod]
    public void HiddenLayer_TryUpdateParams()
    {
      var output = new ScalarProductOutputLayer(2, -3);
      var layer = new MatrixHiddenLayer(1, 1, 1, -1);
      layer.AddNextLayer(output);
      layer.Build();
      var pars = new double[] { 1, 2, 3, -4, -1, 3, 1, -1, 2, -2, 1, 2, 2, 4 };
      var cursor = 1;

      var res = layer.TryUpdateParams(pars, false, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(7,  cursor);
      Assert.AreEqual(2,  layer.Coeffs[0]);
      Assert.AreEqual(3,  layer.Coeffs[1]);
      Assert.AreEqual(-4, layer.Coeffs[2]);
      Assert.AreEqual(-1, layer.Coeffs[3]);
      Assert.AreEqual(3,  output.Coeffs[0]);
      Assert.AreEqual(1,  output.Coeffs[1]);

      res = layer.TryUpdateParams(pars, true, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(13, cursor);
      Assert.AreEqual(1,  layer.Coeffs[0]);
      Assert.AreEqual(5,  layer.Coeffs[1]);
      Assert.AreEqual(-6, layer.Coeffs[2]);
      Assert.AreEqual(0,  layer.Coeffs[3]);
      Assert.AreEqual(5,  output.Coeffs[0]);
      Assert.AreEqual(3,  output.Coeffs[1]);

      res = layer.TryUpdateParams(pars, false, ref cursor);
      Assert.IsFalse(res);
      Assert.AreEqual(13, cursor);
      Assert.AreEqual(1,  layer.Coeffs[0]);
      Assert.AreEqual(5,  layer.Coeffs[1]);
      Assert.AreEqual(-6, layer.Coeffs[2]);
      Assert.AreEqual(0,  layer.Coeffs[3]);
      Assert.AreEqual(5,  output.Coeffs[0]);
      Assert.AreEqual(3,  output.Coeffs[1]);
    }

    #endregion

    #region CompositeLayer

    [TestMethod]
    [ExpectedException(typeof(MLException))]
    public void CompositeLayer_Build_NoNextLayer()
    {
      var layer = new TestCompositeLayer(2, -1);
      var sub1 = new DoublingLayer();
      var sub2 = new ShiftingLayer(3);
      layer.AddSubLayer(sub1);
      layer.AddSubLayer(sub2);

      layer.Build();
    }

    [TestMethod]
    [ExpectedException(typeof(MLException))]
    public void CompositeLayer_Build_NoSubLayers()
    {
      var layer = new TestCompositeLayer(2, -1);
      var output = new IdentityLayer();
      layer.AddNextLayer(output);

      layer.Build();
    }

    [TestMethod]
    public void CompositeLayer_Build()
    {
      var layer = getTestCompositelayer();
      layer.Build();

      Assert.AreEqual(2, layer.ParamsCount);
      Assert.AreEqual(0, layer.SubLayers[0].ParamsCount);
      Assert.AreEqual(1, layer.SubLayers[1].ParamsCount);
      Assert.AreEqual(0, layer.NextLayer.ParamsCount);
    }

    [TestMethod]
    public void CompositeLayer_Calculate()
    {
      var layer = getTestCompositelayer();
      layer.Build();

      var res = (double)layer.Calculate(2);

      Assert.AreEqual(3, res);
    }

    [TestMethod]
    public void CompositeLayer_TryGetParam()
    {
      var layer = getTestCompositelayer();
      layer.Build();

      double par1;
      double par2;
      double par3;
      double par4;
      var res1 = layer.TryGetParam(0, out par1);
      var res2 = layer.TryGetParam(1, out par2);
      var res3 = layer.TryGetParam(2, out par3);
      var res4 = layer.TryGetParam(3, out par4);

      Assert.IsTrue(res1);
      Assert.AreEqual(2, par1);
      Assert.IsTrue(res2);
      Assert.AreEqual(-1, par2);
      Assert.IsTrue(res3);
      Assert.AreEqual(3, par3);
      Assert.IsFalse(res4);
      Assert.AreEqual(0, par4);
    }

    [TestMethod]
    public void CompositeLayer_TrySetParam()
    {
      var layer = getTestCompositelayer();
      layer.Build();

      var res1 = layer.TrySetParam(0,  1, false);
      var res2 = layer.TrySetParam(1,  1, true);
      var res3 = layer.TrySetParam(2, -1, false);
      var res4 = layer.TrySetParam(3, -2, true);

      Assert.IsTrue(res1);
      Assert.AreEqual(1, layer.Coeffs[0]);
      Assert.IsTrue(res2);
      Assert.AreEqual(0, layer.Coeffs[1]);
      Assert.IsTrue(res3);
      Assert.AreEqual(-1, ((ShiftingLayer)layer.SubLayers[1]).Shift);
      Assert.IsFalse(res4);
    }

    [TestMethod]
    public void CompositeLayer_TryUpdateParams()
    {
      var layer = getTestCompositelayer();
      layer.Build();
      var pars = new double[] { 1, 2, 3, -4, -1, 3, 1, -1 };
      var cursor = 1;

      var res = layer.TryUpdateParams(pars, false, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(4,  cursor);
      Assert.AreEqual(2,  layer.Coeffs[0]);
      Assert.AreEqual(3,  layer.Coeffs[1]);
      Assert.AreEqual(-4, ((ShiftingLayer)layer.SubLayers[1]).Shift);

      res = layer.TryUpdateParams(pars, true, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(7, cursor);
      Assert.AreEqual(1,  layer.Coeffs[0]);
      Assert.AreEqual(6,  layer.Coeffs[1]);
      Assert.AreEqual(-3, ((ShiftingLayer)layer.SubLayers[1]).Shift);

      res = layer.TryUpdateParams(pars, false, ref cursor);
      Assert.IsFalse(res);
      Assert.AreEqual(7, cursor);
      Assert.AreEqual(1,  layer.Coeffs[0]);
      Assert.AreEqual(6,  layer.Coeffs[1]);
      Assert.AreEqual(-3, ((ShiftingLayer)layer.SubLayers[1]).Shift);
    }

    #endregion

    #region ComputingNetwork

    [TestMethod]
    [ExpectedException(typeof(MLException))]
    public void ComputingNetwork_Build_NoFirstLayer()
    {
      var network = new ComputingNetwork<Point2D, double>();
      network.Build();
    }

    [TestMethod]
    public void ComputingNetwork_Build()
    {
      var net = getTestNetwork();
      net.Build();

      Assert.AreEqual(0, net.ParamsCount);
      Assert.AreEqual(0, net.BaseLayer.ParamsCount);
      Assert.AreEqual(2, ((dynamic)net).BaseLayer.NextLayer.ParamsCount);
      Assert.AreEqual(0, ((dynamic)net).BaseLayer.NextLayer.SubLayers[0].ParamsCount);
      Assert.AreEqual(1, ((dynamic)net).BaseLayer.NextLayer.SubLayers[0].NextLayer.ParamsCount);
      Assert.AreEqual(1, ((dynamic)net).BaseLayer.NextLayer.SubLayers[1].ParamsCount);
      Assert.AreEqual(1, ((dynamic)net).BaseLayer.NextLayer.NextLayer.ParamsCount);
    }

    [TestMethod]
    public void ComputingNetwork_Calculate()
    {
      var net = getTestNetwork();
      net.Build();

      var input1 = new Point2D(3, 4);
      var input2 = new Point2D(3, 2);
      var res1 = net.Calculate(input1);
      var res2 = net.Calculate(input2);

      Assert.AreEqual(-7, res1);
      Assert.AreEqual(-4, res2);
    }

    [TestMethod]
    public void ComputingNetwork_TryGetParam()
    {
      var net = getTestNetwork();
      net.Build();

      double w1;
      double w2;
      double w3;
      double w4;
      double w5;
      double w6;
      var res1 = net.TryGetParam(0,  out w1);
      var res2 = net.TryGetParam(1,  out w2);
      var res3 = net.TryGetParam(2,  out w3);
      var res4 = net.TryGetParam(3,  out w4);
      var res5 = net.TryGetParam(4,  out w5);
      var res6 = net.TryGetParam(5,  out w6);

      Assert.IsTrue(res1);
      Assert.AreEqual(2, w1);
      Assert.IsTrue(res2);
      Assert.AreEqual(-1, w2);
      Assert.IsTrue(res3);
      Assert.AreEqual(-1, w3);
      Assert.IsTrue(res4);
      Assert.AreEqual(2, w4);
      Assert.IsTrue(res5);
      Assert.AreEqual(3, w5);
      Assert.IsFalse(res6);
      Assert.AreEqual(0, w6);
    }

    [TestMethod]
    public void ComputingNetwork_TrySetParam()
    {
      var net = getTestNetwork();
      net.Build();

      var res1 = net.TrySetParam(0, 1, false);
      var res2 = net.TrySetParam(1, 2, true);
      var res3 = net.TrySetParam(2, 3, false);
      var res4 = net.TrySetParam(3, 4, true);
      var res5 = net.TrySetParam(4, 5, false);
      var res6 = net.TrySetParam(5, 6, true);

      Assert.IsTrue(res1);
      Assert.AreEqual(1, ((dynamic)net).BaseLayer.NextLayer.Coeffs[0]);
      Assert.IsTrue(res2);
      Assert.AreEqual(1, ((dynamic)net).BaseLayer.NextLayer.Coeffs[1]);
      Assert.IsTrue(res3);
      Assert.AreEqual(3, ((dynamic)net).BaseLayer.NextLayer.SubLayers[0].NextLayer.Shift);
      Assert.IsTrue(res4);
      Assert.AreEqual(6, ((dynamic)net).BaseLayer.NextLayer.SubLayers[1].Power);
      Assert.IsTrue(res5);
      Assert.AreEqual(5, ((dynamic)net).BaseLayer.NextLayer.NextLayer.Shift);
      Assert.IsFalse(res6);
    }

    [TestMethod]
    public void ComputingNetwork_TryUpdateParams()
    {
      var net = getTestNetwork();
      net.Build();
      var pars = new double[] { 1, 2, 3, -4, -1, 3, 1, -1, 2, -2, 1, 2 };
      var cursor = 1;

      var res = net.TryUpdateParams(pars, false, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(6,  cursor);
      Assert.AreEqual(2,  ((dynamic)net).BaseLayer.NextLayer.Coeffs[0]);
      Assert.AreEqual(3,  ((dynamic)net).BaseLayer.NextLayer.Coeffs[1]);
      Assert.AreEqual(-4, ((dynamic)net).BaseLayer.NextLayer.SubLayers[0].NextLayer.Shift);
      Assert.AreEqual(-1, ((dynamic)net).BaseLayer.NextLayer.SubLayers[1].Power);
      Assert.AreEqual(3,  ((dynamic)net).BaseLayer.NextLayer.NextLayer.Shift);

      res = net.TryUpdateParams(pars, true, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(11, cursor);
      Assert.AreEqual(3,  ((dynamic)net).BaseLayer.NextLayer.Coeffs[0]);
      Assert.AreEqual(2,  ((dynamic)net).BaseLayer.NextLayer.Coeffs[1]);
      Assert.AreEqual(-2,  ((dynamic)net).BaseLayer.NextLayer.SubLayers[0].NextLayer.Shift);
      Assert.AreEqual(-3, ((dynamic)net).BaseLayer.NextLayer.SubLayers[1].Power);
      Assert.AreEqual(4,  ((dynamic)net).BaseLayer.NextLayer.NextLayer.Shift);

      res = net.TryUpdateParams(pars, false, ref cursor);
      Assert.IsFalse(res);
      Assert.AreEqual(11, cursor);
      Assert.AreEqual(3,  ((dynamic)net).BaseLayer.NextLayer.Coeffs[0]);
      Assert.AreEqual(2,  ((dynamic)net).BaseLayer.NextLayer.Coeffs[1]);
      Assert.AreEqual(-2,  ((dynamic)net).BaseLayer.NextLayer.SubLayers[0].NextLayer.Shift);
      Assert.AreEqual(-3, ((dynamic)net).BaseLayer.NextLayer.SubLayers[1].Power);
      Assert.AreEqual(4,  ((dynamic)net).BaseLayer.NextLayer.NextLayer.Shift);
    }

    #endregion

    #region Large ComputingNetwork

    [TestMethod]
    public void Large_ComputingNetwork_Build()
    {
      var dim = 100;
      var lcount = 10;
      var net = getLargeNetwork(dim, lcount);
      net.Build();

      Assert.AreEqual(0, net.ParamsCount);

      dynamic layer = net.BaseLayer;
      for (int i=0; i<lcount; i++)
      {
        Assert.AreEqual(10000, layer.ParamsCount);
        layer = layer.NextLayer;
      }
      Assert.AreEqual(100, layer.ParamsCount);
    }

    [TestMethod]
    public void Large_ComputingNetwork_TryGetSetParam()
    {
      var dim = 100;
      var lcount = 10;
      var pcount = dim*dim*lcount+dim;
      var net = getLargeNetwork(dim, lcount);
      net.Build();

      // raw

      for (int i=0; i<pcount+10; i++)
      {
        var result = net.TrySetParam(i, i, false);
        if (i<pcount) Assert.IsTrue(result);
        else Assert.IsFalse(result);
      }

      for (int i=0; i<pcount+10; i++)
      {
        double value;
        var result = net.TryGetParam(i, out value);
        if (i<pcount)
        {
          Assert.IsTrue(result);
          Assert.AreEqual(value, i);
        }
        else
          Assert.IsFalse(result);
      }

      // delta

      for (int i=0; i<pcount+10; i++)
      {
        var result = net.TrySetParam(i, i, true);
        if (i<pcount) Assert.IsTrue(result);
        else Assert.IsFalse(result);
      }

      for (int i=0; i<pcount+10; i++)
      {
        double value;
        var result = net.TryGetParam(i, out value);
        if (i<pcount)
        {
          Assert.IsTrue(result);
          Assert.AreEqual(value, 2*i);
        }
        else
          Assert.IsFalse(result);
      }
    }

    [TestMethod]
    public void Large_ComputingNetwork_TryUpdateParams()
    {
      var dim = 100;
      var lcount = 10;
      var pcount = dim*dim*lcount+dim;
      var net = getLargeNetwork(dim, lcount);
      net.Build();

      var pars = new double[pcount];
      for (int i=0; i<pcount; i++)
        pars[i] = i;

      // raw

      int cursor = 0;
      var result = net.TryUpdateParams(pars, false, ref cursor);
      Assert.IsTrue(result);
      Assert.AreEqual(pcount, cursor);

      for (int i=0; i<pcount; i++)
      {
        double value;
        result = net.TryGetParam(i, out value);
        Assert.IsTrue(result);
        Assert.AreEqual(value, i);
      }

      // delta

      cursor = 0;
      result = net.TryUpdateParams(pars, true, ref cursor);
      Assert.IsTrue(result);
      Assert.AreEqual(pcount, cursor);

      for (int i=0; i<pcount; i++)
      {
        double value;
        result = net.TryGetParam(i, out value);
        Assert.IsTrue(result);
        Assert.AreEqual(value, 2*i);
      }
    }

    [TestMethod]
    public void Large_Bench_ComputingNetwork_Calculate()
    {
      var dim = 100;
      var lcount = 10;
      var net = getLargeNetwork(dim, lcount);
      net.Build();

      var input = new double[dim];
      for (int i=0; i<dim; i++)
        input[i] = 0.0001D;

      var timer = new Stopwatch();
      var times = 1000;
      timer.Start();

      for (int i=0; i<times; i++)
        net.Calculate(input);

      timer.Stop();
      Console.WriteLine("Calculation BM: (dim={0} layers={1}): {2}ms",
                        dim,
                        lcount,
                        (float)timer.ElapsedMilliseconds/times);
    }

    [TestMethod]
    public void Large_Bench_ComputingNetwork_BulkSetVSIndexSet()
    {
      var dim = 100;
      var lcount = 10;
      var pcount = dim*dim*lcount+dim;
      var net = getLargeNetwork(dim, lcount);
      net.Build();

      var pars = new double[pcount];
      pars[12345] = 1.5D;
      pars[2345] = 0.5D;
      pars[73450] = 3.5D;

      var timer = new Stopwatch();
      var times = 1000;
      timer.Start();

      for (int i=0; i<times; i++)
      {
        int cursor = 0;
        net.TryUpdateParams(pars, false, ref cursor);
      }

      timer.Stop();
      Console.WriteLine("Bulk Set BM: (dim={0} layers={1}): {2} ticks",
                        dim,
                        lcount,
                        (float)timer.ElapsedTicks/times);

      timer.Restart();
      for (int i=0; i<times; i++)
      {
        net.TrySetParam(12345, 1.5D, false);
        net.TrySetParam(2345, 0.5D, false);
        net.TrySetParam(73450, 3.5D, false);
      }

      timer.Stop();
      Console.WriteLine("Index Set BM: (dim={0} layers={1}): {2} ticks",
                        dim,
                        lcount,
                        (float)timer.ElapsedTicks/times);
    }

    #endregion

    #region .pvt

    /// <summary>
    ///      Double(x -> 2*x)
    ///     /                \
    /// x ->                  Sc.pr.(w1, w2) -> Identity(x -> x)
    ///     \                /
    ///      Shift(x -> x+w3)
    ///
    ///  params: [w1, w2, w3]
    /// </summary>
    private TestCompositeLayer getTestCompositelayer()
    {
      var layer = new TestCompositeLayer(2, -1);
      var sub1 = new DoublingLayer();
      var sub2 = new ShiftingLayer(3);
      var output = new IdentityLayer();
      layer.AddSubLayer(sub1);
      layer.AddSubLayer(sub2);
      layer.AddNextLayer(output);

      return layer;
    }

    /// <summary>
    ///                       DoubleStep(x -> 2*(int)(x/2)) -> Shift(x -> x+w3)
    ///                      /                                                 \
    ///  (x,y) -> max(x,y) ->                                                   Sc.pr.(w1, w2) -> Shift(x -> x+w5)
    ///                      \                                                 /
    ///                                         Pow(x -> x^w4)
    ///  params: [w1, w2, w3, w4, w5]
    /// </summary>
    private ComputingNetwork<Point2D, double> getTestNetwork()
    {
      var net = new ComputingNetwork<Point2D, double>();
      var max = new MaxLayer();
      net.AddBaseLayer(max);

      var comp = new TestCompositeLayer(2, -1);
      var upper = new DoubleStepLayer();
      upper.AddNextLayer(new ShiftingLayer(-1));
      comp.AddSubLayer(upper);
      var lower = new PowerLayer(2);
      comp.AddSubLayer(lower);
      max.AddNextLayer(comp);

      comp.AddNextLayer(new ShiftingLayer(3));

      return net;
    }

    /// <summary>
    /// (x1,...,x10) -> 10 of 10x10 matrix transform -> scalar product
    ///
    /// params: [w[1,1,1], ..., w[10,10,1], ..., w[10,10,10], w1, w2, ..., w10] - 1010
    /// </summary>
    private ComputingNetwork<double[], double> getLargeNetwork(int dim, int layers)
    {
      var net = new ComputingNetwork<double[], double>();

      var M = new double[dim, dim];
      for (int i=0; i<dim; i++)
      for (int j=0; j<dim; j++)
        M[i,j] = (double)(i+j)/dim;

      var next = new NMatrixLayer(M);
      net.AddBaseLayer(next);

      IHiddenLayer<double[], double[]> layer = next;
      for (int l=0; l<layers-1; l++)
      {
        next = new NMatrixLayer(M);
        layer.AddNextLayer(next);
        layer = next;
      }

      var coeffs = new double[dim];
      for (int i=1; i<dim; i++)
        coeffs[i] = 1;

      layer.AddNextLayer(new ScalarProductOutputLayer(coeffs));

      return net;
    }

    #endregion
  }
}
