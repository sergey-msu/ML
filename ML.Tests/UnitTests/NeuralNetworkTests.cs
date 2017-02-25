using System;
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ML.Core;
using ML.Core.ComputingNetworks;
using ML.NeuralMethods.Model;

namespace ML.Tests.UnitTests
{
  [TestClass]
  public class NeuralNetworkTests : TestBase
  {
    public const double EPS = 0.0000001D;

    #region Inner

    public class SimpleLayer<TNeuron> : NeuralLayer
      where TNeuron : Neuron
    {
      public SimpleLayer(bool useBias) : base(4)
      {
        UseBias = useBias;

        var n1 = this.CreateNeuron<TNeuron>();
        n1[0] = 1;
        n1[1] = -1;
        if (useBias) n1[4] = 2;

        var n2 = this.CreateNeuron<TNeuron>();
        n2[0] = 2;
        n2[2] = 3;
        if (useBias) n2[4] = 1;

        var n3 = this.CreateNeuron<TNeuron>();
        n3[3] = -2;
        if (useBias) n3[4] = -1;
      }
    }

    public class SimpleSparseLayer : SimpleLayer<SparseNeuron>
    {
      public SimpleSparseLayer(bool useBias)
        : base(useBias)
      {
      }
    }

    public class SimpleFullLayer : SimpleLayer<FullNeuron>
    {
      public SimpleFullLayer(bool useBias)
        : base(useBias)
      {
      }
    }

    public class SimpleNetwork<TNeuron> : NeuralNetwork
      where TNeuron : Neuron
    {
      public SimpleNetwork()
      {
        InputDim = 3;

        var l1 = this.CreateLayer();
        var n11 = l1.CreateNeuron<TNeuron>();
        n11[0] = -1;
        var n12 = l1.CreateNeuron<TNeuron>();
        n12[1] = 2;
        n12[2] = 3;
        var n13 = l1.CreateNeuron<TNeuron>();
        n13[2] = -2;

        var l2 = this.CreateLayer();
        var n21 = l2.CreateNeuron<TNeuron>();
        n21[0] = 1;
        n21[1] = -1;
        var n22 = l2.CreateNeuron<TNeuron>();
        n22[2] = 3;

        var output = this.CreateLayer();
        var on = output.CreateNeuron<TNeuron>();
        on[0] = 1;
        on[1] = -1;
      }
    }

    public class LargeNetwork : NeuralNetwork
    {
      public LargeNetwork(int lcount, int ncount)
      {
        InputDim = ncount;

        for (int i = 0; i < lcount; i++)
        {
          var layer = this.CreateLayer();
          for (int j = 0; j < ncount; j++)
          {
            var n = layer.CreateNeuron<FullNeuron>();
            for (int k = 0; k < ncount; k++)
              n[k] = ((double)(i + j + k)) / (ncount * ncount);
          }
        }

        var output = this.CreateLayer();
        var on = output.CreateNeuron<SparseNeuron>();
        for (int i = 0; i < ncount; i++)
          on[i] = (i % 2 == 0) ? 1 : -1;
      }
    }

    #endregion

    [ClassInitialize]
    public static void ClassInit(TestContext context)
    {
      BaseClassInit(context);
    }

    #region SparseNeuron

    [TestMethod]
    public void SparseNeuron_Build()
    {
      var n = new SparseNeuron(4, false);
      n[0] = 0.1D;
      n[2] = -0.2D;
      n[3] = 0.3D;
      n.Build();

      Assert.AreEqual(3, n.ParamCount);
    }

    [TestMethod]
    public void SparseNeuron_UseBias_Build()
    {
      var n = new SparseNeuron(4, true);
      n[0] = 0.1D;
      n[2] = -0.2D;
      n[3] = 0.3D;
      n[4] = 2.5D;
      n.Build();

      Assert.AreEqual(4, n.ParamCount);
    }

    [TestMethod]
    [ExpectedException(typeof(MLException))]
    public void SparseNeuron_Calculate_WrongInput()
    {
      var n = new SparseNeuron(4, false);
      n[0] = 0.1D;
      n[2] = -0.2D;
      n[3] = 0.3D;
      n.Build();
      var input = new double[] { 1, 2, 3, 4, 5 };

      n.Calculate(input);
    }

    [TestMethod]
    public void SparseNeuron_Calculate()
    {
      var n = new SparseNeuron(4, false);
      n[0] = 0.1D;
      n[2] = -0.2D;
      n[3] = 0.3D;
      n.Build();
      var input = new double[] { 1, 2, 3, 4 };

      var result = n.Calculate(input);

      MLAssert.AreEpsEqual(0.7D, result, EPS);
    }

    [TestMethod]
    public void SparseNeuron_UseBias_Calculate()
    {
      var n = new SparseNeuron(4, true);
      n[0] = 0.1D;
      n[2] = -0.2D;
      n[3] = 0.3D;
      n[4] = 0.5D;
      n.Build();
      var input = new double[] { 1, 2, 3, 4 };

      var result = n.Calculate(input);

      MLAssert.AreEpsEqual(1.2D, result, EPS);
    }

    [TestMethod]
    public void SparseNeuron_Calculate_ActivationFunction()
    {
      var n = new SparseNeuron(4, false);
      n[0] = 0.1D;
      n[2] = -0.2D;
      n[3] = 0.3D;
      n.ActivationFunction = Registry.ActivationFunctions.Exp;
      n.Build();
      var input = new double[] { 1, 2, 3, 4 };

      var result = n.Calculate(input);

      MLAssert.AreEpsEqual(Math.Exp(0.7D), result, EPS);
    }

    [TestMethod]
    public void SparseNeuron_Indexer()
    {
      var n = new SparseNeuron(4, false);
      n[0] = 0.1D;
      n[2] = -0.2D;
      n[3] = 0.3D;
      n.Build();

      n.RemoveWeight(0);
      n[1] = 0.5D;
      n[2] = 0.6D;

      Assert.AreEqual(0,    n[0]);
      Assert.AreEqual(0.5D, n[1]);
      Assert.AreEqual(0.6D, n[2]);
      Assert.AreEqual(0.3D, n[3]);
    }

    [TestMethod]
    public void SparseNeuron_UseBias_Indexer()
    {
      var n = new SparseNeuron(4, true);
      n[0] =  0.1D;
      n[2] = -0.2D;
      n[3] =  0.3D;
      n[4] =  0.5D;
      n.Build();

      n.RemoveWeight(0);
      n[1] = 0.5D;
      n[2] = 0.6D;

      Assert.AreEqual(0,    n[0]);
      Assert.AreEqual(0.5D, n[1]);
      Assert.AreEqual(0.6D, n[2]);
      Assert.AreEqual(0.3D, n[3]);
      Assert.AreEqual(0.5D, n[4]);
    }

    [TestMethod]
    public void SparseNeuron_TryGetParam()
    {
      var n = new SparseNeuron(4, false);
      n[0] = 0.1D;
      n[2] = -0.2D;
      n[3] = 0.3D;
      n.Build();

      double par1;
      double par2;
      double par3;
      double par4;
      var res1 = n.TryGetParam(0, out par1);
      var res2 = n.TryGetParam(1, out par2);
      var res3 = n.TryGetParam(2, out par3);
      var res4 = n.TryGetParam(3, out par4);

      Assert.IsTrue(res1);
      Assert.AreEqual(0.1D, par1);
      Assert.IsTrue(res2);
      Assert.AreEqual(-0.2D, par2);
      Assert.IsTrue(res3);
      Assert.AreEqual(0.3D, par3);
      Assert.IsFalse(res4);
    }

    [TestMethod]
    public void SparseNeuron_UseBias_TryGetParam()
    {
      var n = new SparseNeuron(4, true);
      n[0] = 0.1D;
      n[2] = -0.2D;
      n[3] = 0.3D;
      n[4] = 0.5D;
      n.Build();

      double par1;
      double par2;
      double par3;
      double par4;
      double par5;
      var res1 = n.TryGetParam(0, out par1);
      var res2 = n.TryGetParam(1, out par2);
      var res3 = n.TryGetParam(2, out par3);
      var res4 = n.TryGetParam(3, out par4);
      var res5 = n.TryGetParam(4, out par5);

      Assert.IsTrue(res1);
      Assert.AreEqual(0.1D, par1);
      Assert.IsTrue(res2);
      Assert.AreEqual(-0.2D, par2);
      Assert.IsTrue(res3);
      Assert.AreEqual(0.3D, par3);
      Assert.IsTrue(res4);
      Assert.AreEqual(0.5D, par4);
      Assert.IsFalse(res5);
    }

    [TestMethod]
    public void SparseNeuron_TrySetParam()
    {
      var n = new SparseNeuron(4, false);
      n[0] = 0.1D;
      n[2] = -0.2D;
      n[3] = 0.3D;
      n.Build();

      var res1 = n.TrySetParam(0, -1, false);
      var res2 = n.TrySetParam(1, 1, true);
      var res3 = n.TrySetParam(2, 2, false);
      var res4 = n.TrySetParam(3, 3, true);

      Assert.IsTrue(res1);
      Assert.AreEqual(-1, n[0]);
      Assert.IsTrue(res2);
      Assert.AreEqual(0.8D, n[2]);
      Assert.IsTrue(res3);
      Assert.AreEqual(2, n[3]);
      Assert.IsFalse(res4);
    }

    [TestMethod]
    public void SparseNeuron_UseBias_TrySetParam()
    {
      var n = new SparseNeuron(4, true);
      n[0] = 0.1D;
      n[2] = -0.2D;
      n[3] = 0.3D;
      n[4] = 0.5D;
      n.Build();

      var res1 = n.TrySetParam(0, -1, false);
      var res2 = n.TrySetParam(1,  1, true);
      var res3 = n.TrySetParam(2,  2, false);
      var res4 = n.TrySetParam(3,  4, true);
      var res5 = n.TrySetParam(4,  5, true);

      Assert.IsTrue(res1);
      Assert.AreEqual(-1, n[0]);
      Assert.IsTrue(res2);
      Assert.AreEqual(0.8D, n[2]);
      Assert.IsTrue(res3);
      Assert.AreEqual(2, n[3]);
      Assert.IsTrue(res3);
      Assert.AreEqual(4.5, n[4]);
      Assert.IsFalse(res5);
    }

    [TestMethod]
    public void SparseNeuron_TryUpdateParams()
    {
      var n = new SparseNeuron(4, false);
      n[0] = 0.1D;
      n[2] = -0.2D;
      n[3] = 0.3D;
      n.Build();

      var pars = new double[] { 1, 2, -1, 3, 1, 2, -3, -1, 1 };
      var cursor = 1;

      var res = n.TryUpdateParams(pars, false, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(4, cursor);
      Assert.AreEqual(2, n[0]);
      Assert.AreEqual(-1, n[2]);
      Assert.AreEqual(3, n[3]);

      res = n.TryUpdateParams(pars, true, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(7, cursor);
      Assert.AreEqual(3, n[0]);
      Assert.AreEqual(1, n[2]);
      Assert.AreEqual(0, n[3]);

      res = n.TryUpdateParams(pars, false, ref cursor);
      Assert.IsFalse(res);
    }

    [TestMethod]
    public void SparseNeuron_UseBias_TryUpdateParams()
    {
      var n = new SparseNeuron(4, true);
      n[0] = 0.1D;
      n[2] = -0.2D;
      n[3] = 0.3D;
      n[4] = 0.5D;
      n.Build();

      var pars = new double[] { 1, 2, -1, 3, 1, 2, -3, -1, 1, 2, -2 };
      var cursor = 1;

      var res = n.TryUpdateParams(pars, false, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(5,  cursor);
      Assert.AreEqual(2,  n[0]);
      Assert.AreEqual(-1, n[2]);
      Assert.AreEqual(3,  n[3]);
      Assert.AreEqual(1,  n[4]);

      res = n.TryUpdateParams(pars, true, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(9, cursor);
      Assert.AreEqual(4,  n[0]);
      Assert.AreEqual(-4, n[2]);
      Assert.AreEqual(2,  n[3]);
      Assert.AreEqual(2,  n[4]);

      res = n.TryUpdateParams(pars, false, ref cursor);
      Assert.IsFalse(res);
    }

    [TestMethod]
    public void SparseNeuron_ComplexCalculation()
    {
      var cinput = 11;
      var n = new SparseNeuron(cinput, false);
      var input = new double[cinput];
      for (int i = 0; i < cinput; i++)
      {
        if (i % 2 == 0) n[i] = i;
        input[i] = 1;
      }
      n.Build();

      var result = n.Calculate(input);
      Assert.AreEqual(30, result);

      var success = n.TrySetParam(0, 1, false);
      Assert.IsTrue(success);
      success = n.TrySetParam(1, -2, true);
      Assert.IsTrue(success);

      result = n.Calculate(input);
      Assert.AreEqual(29, result);

      var pars = new double[n.ParamCount];
      for (int i = 0; i < n.ParamCount; i++)
        pars[i] = 1;
      var cursor = 0;
      n.TryUpdateParams(pars, true, ref cursor);

      result = n.Calculate(input);
      Assert.AreEqual(35, result);

      cursor = 0;
      n.TryUpdateParams(pars, false, ref cursor);

      result = n.Calculate(input);
      Assert.AreEqual(6, result);
    }

    [TestMethod]
    public void SparseNeuron_Bench_Calculate()
    {
      int cinput = 1000000;
      var n = new SparseNeuron(cinput, false);
      var input = new double[cinput];
      for (int i = 0; i < cinput; i++)
      {
        if (i % 9 != 0) n[i] = ((double)(i % 1234)) / 1000;
        input[i] = ((double)i) / 10000;
      }
      n.Build();

      var times = 200;
      var timer = new Stopwatch();
      timer.Start();
      for (int i = 0; i < times; i++)
        n.Calculate(input);
      timer.Stop();

      Console.WriteLine("Neuron Calculate BM: cinput={0} elapsed={1}ms", cinput, (int)(timer.Elapsed.TotalMilliseconds / times));
    }

    [TestMethod]
    public void SparseNeuron_Bench_TryUpdateParams_Calculate()
    {
      int cinput = 1000000;
      var n = new SparseNeuron(cinput, false);
      for (int i = 0; i < cinput; i++)
      {
        if (i % 9 != 0) n[i] = ((double)(i % 1234)) / 1000;
      }
      n.Build();
      var pars = new double[n.ParamCount];
      for (int i = 0; i < pars.Length; i++)
        pars[i] = ((double)i) / 10000;

      var times = 200;
      var timer = new Stopwatch();
      timer.Start();
      for (int i = 0; i < times; i++)
      {
        var cursor = 0;
        var res = n.TryUpdateParams(pars, true, ref cursor);
        if (!res) throw new MLCorruptedIndexException();
      }
      timer.Stop();

      Console.WriteLine("Neuron update params BM: cinput={0} elapsed={1}ms", cinput, (int)(timer.Elapsed.TotalMilliseconds / times));
    }

    [TestMethod]
    public void SparseNeuron_Bench_BulkSetVSIndexSet()
    {
      int cinput = 1000000;
      var n = new SparseNeuron(cinput, false);
      for (int i = 0; i < cinput; i++)
      {
        if (i % 9 != 0) n[i] = ((double)(i % 1234)) / 1000;
      }
      n.Build();

      var pars = new double[n.ParamCount];
      pars[1234] = 1.0D;
      pars[123404] = -1.0D;
      pars[200404] = -2.0D;
      pars[700404] = 3.0D;

      var times = 200;
      var timer = new Stopwatch();
      timer.Start();
      for (int i = 0; i < times; i++)
      {
        var cursor = 0;
        n.TryUpdateParams(pars, true, ref cursor);
      }
      timer.Stop();

      Console.WriteLine("Neuron Bulk set BM: cinput={0} elapsed={1}ms", cinput, (int)(timer.Elapsed.TotalMilliseconds / times));

      timer.Reset();
      timer.Start();
      for (int i = 0; i < times; i++)
      {
        n.TrySetParam(1234, 1.0D, true);
        n.TrySetParam(123404, -1.0D, true);
        n.TrySetParam(200404, -2.0D, true);
        n.TrySetParam(700404, 3.0D, true);
      }
      timer.Stop();

      Console.WriteLine("Neuron Index set BM: cinput={0} elapsed={1}ms", cinput, (int)(timer.Elapsed.TotalMilliseconds / times));
    }

    #endregion

    #region FullNeuron

    [TestMethod]
    public void FullNeuron_Build()
    {
      var n = new FullNeuron(4, false);
      n[0] = 0.1D;
      n[2] = -0.2D;
      n[3] = 0.3D;
      n.Build();

      Assert.AreEqual(4, n.ParamCount);
    }

    [TestMethod]
    public void FullNeuron_UseBias_Build()
    {
      var n = new FullNeuron(4, true);
      n[0] = 0.1D;
      n[2] = -0.2D;
      n[3] = 0.3D;
      n[4] = 0.5D;
      n.Build();

      Assert.AreEqual(5, n.ParamCount);
    }

    [TestMethod]
    [ExpectedException(typeof(MLException))]
    public void FullNeuron_Calculate_WrongInput()
    {
      var n = new FullNeuron(4, false);
      n[0] = 0.1D;
      n[2] = -0.2D;
      n[3] = 0.3D;
      n.Build();
      var input = new double[] { 1, 2, 3, 4, 5 };

      n.Calculate(input);
    }

    [TestMethod]
    public void FullNeuron_Calculate()
    {
      var n = new FullNeuron(4, false);
      n[0] = 0.1D;
      n[2] = -0.2D;
      n[3] = 0.3D;
      n.Build();
      var input = new double[] { 1, 2, 3, 4 };

      var result = n.Calculate(input);

      MLAssert.AreEpsEqual(0.7D, result, EPS);
    }

    [TestMethod]
    public void FullNeuron_UseBias_Calculate()
    {
      var n = new FullNeuron(4, true);
      n[0] = 0.1D;
      n[2] = -0.2D;
      n[3] = 0.3D;
      n[4] = 0.5;
      n.Build();
      var input = new double[] { 1, 2, 3, 4 };

      var result = n.Calculate(input);

      MLAssert.AreEpsEqual(1.2D, result, EPS);
    }

    [TestMethod]
    public void FullNeuron_Calculate_ActivationFunction()
    {
      var n = new FullNeuron(4, false);
      n[0] = 0.1D;
      n[2] = -0.2D;
      n[3] = 0.3D;
      n.ActivationFunction = Registry.ActivationFunctions.Exp;
      n.Build();
      var input = new double[] { 1, 2, 3, 4 };

      var result = n.Calculate(input);

      MLAssert.AreEpsEqual(Math.Exp(0.7D), result, EPS);
    }

    [TestMethod]
    public void FullNeuron_Indexer()
    {
      var n = new FullNeuron(4, false);
      n[0] = 0.1D;
      n[2] = -0.2D;
      n[3] = 0.3D;
      n.Build();

      Assert.AreEqual(0.1D,  n[0]);
      Assert.AreEqual(0,     n[1]);
      Assert.AreEqual(-0.2D, n[2]);
      Assert.AreEqual(0.3D,  n[3]);
    }

    [TestMethod]
    public void FullNeuron_UseBias_Indexer()
    {
      var n = new FullNeuron(4, true);
      n[0] = 0.1D;
      n[2] = -0.2D;
      n[3] = 0.3D;
      n[4] = 0.5D;
      n.Build();

      Assert.AreEqual(0.1D,  n[0]);
      Assert.AreEqual(0,     n[1]);
      Assert.AreEqual(-0.2D, n[2]);
      Assert.AreEqual(0.3D,  n[3]);
      Assert.AreEqual(0.5D,  n[4]);
    }

    [TestMethod]
    public void FullNeuron_TryGetParam()
    {
      var n = new FullNeuron(4, false);
      n[0] = 0.1D;
      n[2] = -0.2D;
      n[3] = 0.3D;
      n.Build();

      double par1;
      double par2;
      double par3;
      double par4;
      var res1 = n.TryGetParam(0, out par1);
      var res2 = n.TryGetParam(1, out par2);
      var res3 = n.TryGetParam(2, out par3);
      var res4 = n.TryGetParam(3, out par4);

      Assert.IsTrue(res1);
      Assert.IsTrue(res2);
      Assert.IsTrue(res3);
      Assert.IsTrue(res4);
      Assert.AreEqual(0.1D, par1);
      Assert.AreEqual(0, par2);
      Assert.AreEqual(-0.2D, par3);
      Assert.AreEqual(0.3D, par4);
    }

    [TestMethod]
    public void FullNeuron_UseBias_TryGetParam()
    {
      var n = new FullNeuron(4, true);
      n[0] = 0.1D;
      n[2] = -0.2D;
      n[3] = 0.3D;
      n[4] = 0.5D;
      n.Build();

      double par1;
      double par2;
      double par3;
      double par4;
      double par5;
      double par6;
      var res1 = n.TryGetParam(0, out par1);
      var res2 = n.TryGetParam(1, out par2);
      var res3 = n.TryGetParam(2, out par3);
      var res4 = n.TryGetParam(3, out par4);
      var res5 = n.TryGetParam(4, out par5);
      var res6 = n.TryGetParam(5, out par6);

      Assert.IsTrue(res1);
      Assert.IsTrue(res2);
      Assert.IsTrue(res3);
      Assert.IsTrue(res4);
      Assert.IsTrue(res5);
      Assert.IsFalse(res6);
      Assert.AreEqual(0.1D, par1);
      Assert.AreEqual(0, par2);
      Assert.AreEqual(-0.2D, par3);
      Assert.AreEqual(0.3D, par4);
      Assert.AreEqual(0.5D, par5);
    }

    [TestMethod]
    public void FullNeuron_TrySetParam()
    {
      var n = new FullNeuron(4, false);
      n[0] = 0.1D;
      n[2] = -0.2D;
      n[3] = 0.3D;
      n.Build();

      var res1 = n.TrySetParam(0, -1, false);
      var res2 = n.TrySetParam(1,  1, true);
      var res3 = n.TrySetParam(2,  2, false);
      var res4 = n.TrySetParam(3,  3, true);

      Assert.IsTrue(res1);
      Assert.IsTrue(res2);
      Assert.IsTrue(res3);
      Assert.IsTrue(res4);
      Assert.AreEqual(-1, n[0]);
      Assert.AreEqual(1, n[1]);
      Assert.AreEqual(2, n[2]);
      Assert.AreEqual(3.3D, n[3]);
    }

    [TestMethod]
    public void FullNeuron_UseBias_TrySetParam()
    {
      var n = new FullNeuron(4, true);
      n[0] = 0.1D;
      n[2] = -0.2D;
      n[3] = 0.3D;
      n[4] = 0.5D;
      n.Build();

      var res1 = n.TrySetParam(0, -1, false);
      var res2 = n.TrySetParam(1,  1, true);
      var res3 = n.TrySetParam(2,  2, false);
      var res4 = n.TrySetParam(3,  3, true);
      var res5 = n.TrySetParam(4,  4, true);
      var res6 = n.TrySetParam(5,  5, true);

      Assert.IsTrue(res1);
      Assert.IsTrue(res2);
      Assert.IsTrue(res3);
      Assert.IsTrue(res4);
      Assert.IsTrue(res5);
      Assert.IsFalse(res6);
      Assert.AreEqual(-1,   n[0]);
      Assert.AreEqual(1,    n[1]);
      Assert.AreEqual(2,    n[2]);
      Assert.AreEqual(3.3D, n[3]);
      Assert.AreEqual(4.5D, n[4]);
    }

    [TestMethod]
    public void FullNeuron_TryUpdateParams()
    {
      var n = new FullNeuron(4, false);
      n[0] = 0.1D;
      n[2] = -0.2D;
      n[3] = 0.3D;
      n.Build();

      var pars = new double[] { 1, 2, -1, 3, 1, 2, -3, -1, 1, 7 };
      var cursor = 1;

      var res = n.TryUpdateParams(pars, false, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(5, cursor);
      Assert.AreEqual(2,  n[0]);
      Assert.AreEqual(-1, n[1]);
      Assert.AreEqual(3,  n[2]);
      Assert.AreEqual(1,  n[3]);

      res = n.TryUpdateParams(pars, true, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(9, cursor);
      Assert.AreEqual(4,  n[0]);
      Assert.AreEqual(-4, n[1]);
      Assert.AreEqual(2,  n[2]);
      Assert.AreEqual(2, n[3]);

      res = n.TryUpdateParams(pars, false, ref cursor);
      Assert.IsFalse(res);
    }

    [TestMethod]
    public void FullNeuron_UseBias_TryUpdateParams()
    {
      var n = new FullNeuron(4, true);
      n[0] = 0.1D;
      n[2] = -0.2D;
      n[3] = 0.3D;
      n[4] = 0.5D;
      n.Build();

      var pars = new double[] { 1, 2, -1, 3, 1, 2, -3, -1, 1, 7, -1, 1 };
      var cursor = 1;

      var res = n.TryUpdateParams(pars, false, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(6, cursor);
      Assert.AreEqual(2,  n[0]);
      Assert.AreEqual(-1, n[1]);
      Assert.AreEqual(3,  n[2]);
      Assert.AreEqual(1,  n[3]);
      Assert.AreEqual(2,  n[4]);

      res = n.TryUpdateParams(pars, true, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(11, cursor);
      Assert.AreEqual(-1,  n[0]);
      Assert.AreEqual(-2,  n[1]);
      Assert.AreEqual(4,   n[2]);
      Assert.AreEqual(8,   n[3]);
      Assert.AreEqual(1,   n[4]);

      res = n.TryUpdateParams(pars, false, ref cursor);
      Assert.IsFalse(res);
    }

    [TestMethod]
    public void FullNeuron_ComplexCalculation()
    {
      var cinput = 11;
      var n = new FullNeuron(cinput, false);
      var input = new double[cinput];
      for (int i = 0; i < cinput; i++)
      {
        if (i % 2 == 0) n[i] = i;
        input[i] = 1;
      }
      n.Build();

      var result = n.Calculate(input);
      Assert.AreEqual(30, result);

      var success = n.TrySetParam(0, 1, false);
      Assert.IsTrue(success);
      success = n.TrySetParam(1, -2, true);
      Assert.IsTrue(success);

      result = n.Calculate(input);
      Assert.AreEqual(29, result);

      var pars = new double[n.ParamCount];
      for (int i = 0; i < n.ParamCount; i++)
        pars[i] = 1;
      var cursor = 0;
      n.TryUpdateParams(pars, true, ref cursor);

      result = n.Calculate(input);
      Assert.AreEqual(40, result);

      cursor = 0;
      n.TryUpdateParams(pars, false, ref cursor);

      result = n.Calculate(input);
      Assert.AreEqual(11, result);
    }

    [TestMethod]
    public void FullNeuron_Bench_Calculate()
    {
      int cinput = 1000000;
      var n = new FullNeuron(cinput, false);
      var input = new double[cinput];
      for (int i = 0; i < cinput; i++)
      {
        if (i % 9 != 0) n[i] = ((double)(i % 1234)) / 1000;
        input[i] = ((double)i) / 10000;
      }
      n.Build();

      var times = 200;
      var timer = new Stopwatch();
      timer.Start();
      for (int i = 0; i < times; i++)
        n.Calculate(input);
      timer.Stop();

      Console.WriteLine("Neuron Calculate BM: cinput={0} elapsed={1}ms", cinput, (int)(timer.Elapsed.TotalMilliseconds / times));
    }

    [TestMethod]
    public void FullNeuron_Bench_TryUpdateParams_Calculate()
    {
      int cinput = 1000000;
      var n = new FullNeuron(cinput, false);
      for (int i = 0; i < cinput; i++)
      {
        if (i % 9 != 0) n[i] = ((double)(i % 1234)) / 1000;
      }
      n.Build();
      var pars = new double[n.ParamCount];
      for (int i = 0; i < pars.Length; i++)
        pars[i] = ((double)i) / 10000;

      var times = 200;
      var timer = new Stopwatch();
      timer.Start();
      for (int i = 0; i < times; i++)
      {
        var cursor = 0;
        var res = n.TryUpdateParams(pars, true, ref cursor);
        if (!res) throw new MLCorruptedIndexException();
      }
      timer.Stop();

      Console.WriteLine("Neuron update params BM: cinput={0} elapsed={1}ms", cinput, (int)(timer.Elapsed.TotalMilliseconds / times));
    }

    [TestMethod]
    public void FullNeuron_Bench_BulkSetVSIndexSet()
    {
      int cinput = 1000000;
      var n = new FullNeuron(cinput, false);
      for (int i = 0; i < cinput; i++)
      {
        if (i % 9 != 0) n[i] = ((double)(i % 1234)) / 1000;
      }
      n.Build();

      var pars = new double[n.ParamCount];
      pars[1234] = 1.0D;
      pars[123404] = -1.0D;
      pars[200404] = -2.0D;
      pars[700404] = 3.0D;

      var times = 200;
      var timer = new Stopwatch();
      timer.Start();
      for (int i = 0; i < times; i++)
      {
        var cursor = 0;
        n.TryUpdateParams(pars, true, ref cursor);
      }
      timer.Stop();

      Console.WriteLine("Neuron Bulk set BM: cinput={0} elapsed={1}ms", cinput, (int)(timer.Elapsed.TotalMilliseconds / times));

      timer.Reset();
      timer.Start();
      for (int i = 0; i < times; i++)
      {
        n.TrySetParam(1234, 1.0D, true);
        n.TrySetParam(123404, -1.0D, true);
        n.TrySetParam(200404, -2.0D, true);
        n.TrySetParam(700404, 3.0D, true);
      }
      timer.Stop();

      Console.WriteLine("Neuron Index set BM: cinput={0} elapsed={1}ms", cinput, (int)(timer.Elapsed.TotalMilliseconds / times));
    }

    #endregion

    #region NeuralLayer

    [TestMethod]
    public void NeuralLayer_AddNeuron()
    {
      var layer = new NeuralLayer(12);
      var n1 = layer.CreateNeuron<SparseNeuron>();

      Assert.IsNotNull(n1);
    }

    [TestMethod]
    public void NeuralLayer_Build()
    {
      var layer = new SimpleSparseLayer(false);
      layer.Build();

      Assert.AreEqual(3, layer.SubNodes.Length);
      Assert.AreEqual(2, layer[0].ParamCount);
      Assert.AreEqual(2, layer[1].ParamCount);
      Assert.AreEqual(1, layer[2].ParamCount);
    }

    [TestMethod]
    public void NeuralLayer_Calculate()
    {
      var layer = new SimpleSparseLayer(false);
      layer.Build();

      var res1 = layer.Calculate(new double[] { 1, 1, 1, 1 });
      Assert.AreEqual(3, res1.Length);
      Assert.AreEqual(0, res1[0]);
      Assert.AreEqual(5, res1[1]);
      Assert.AreEqual(-2, res1[2]);

      var res2 = layer.Calculate(new double[] { 1, -1, 1, -1 });
      Assert.AreEqual(3, res2.Length);
      Assert.AreEqual(2, res2[0]);
      Assert.AreEqual(5, res2[1]);
      Assert.AreEqual(2, res2[2]);
    }

    [TestMethod]
    public void NeuralLayer_TryGetParam()
    {
      var layer = new SimpleSparseLayer(false);
      layer.Build();

      double par1;
      double par2;
      double par3;
      double par4;
      double par5;
      double par6;
      var res1 = layer.TryGetParam(0, out par1);
      var res2 = layer.TryGetParam(1, out par2);
      var res3 = layer.TryGetParam(2, out par3);
      var res4 = layer.TryGetParam(3, out par4);
      var res5 = layer.TryGetParam(4, out par5);
      var res6 = layer.TryGetParam(5, out par6);

      Assert.IsTrue(res1);
      Assert.AreEqual(1, par1);
      Assert.IsTrue(res2);
      Assert.AreEqual(-1, par2);
      Assert.IsTrue(res3);
      Assert.AreEqual(2, par3);
      Assert.IsTrue(res4);
      Assert.AreEqual(3, par4);
      Assert.IsTrue(res5);
      Assert.AreEqual(-2, par5);
      Assert.IsFalse(res6);
      Assert.AreEqual(0, par6);
    }

    [TestMethod]
    public void NeuralLayer_TrySetParam()
    {
      var layer = new SimpleSparseLayer(false);
      layer.Build();

      var res1 = layer.TrySetParam(0, 1, false);
      var res2 = layer.TrySetParam(1, 1, true);
      var res3 = layer.TrySetParam(2, -2, false);
      var res4 = layer.TrySetParam(3, -4, true);
      var res5 = layer.TrySetParam(4, 4, false);
      var res6 = layer.TrySetParam(5, 4, false);

      Assert.IsTrue(res1);
      Assert.AreEqual(1, layer[0][0]);
      Assert.IsTrue(res2);
      Assert.AreEqual(0, layer[0][1]);
      Assert.IsTrue(res3);
      Assert.AreEqual(-2, layer[1][0]);
      Assert.IsTrue(res4);
      Assert.AreEqual(-1, layer[1][2]);
      Assert.IsTrue(res5);
      Assert.AreEqual(4, layer[2][3]);
      Assert.IsFalse(res6);
    }

    [TestMethod]
    public void NeuralLayer_TryUpdateParams()
    {
      var layer = new SimpleSparseLayer(false);
      layer.Build();

      var pars = new double[] { 1, 2, 3, -4, -1, 3, 1, -1, 2, 5, 3, 5 };
      var cursor = 1;

      var res = layer.TryUpdateParams(pars, false, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(6, cursor);
      Assert.AreEqual(2, layer[0][0]);
      Assert.AreEqual(3, layer[0][1]);
      Assert.AreEqual(-4, layer[1][0]);
      Assert.AreEqual(-1, layer[1][2]);
      Assert.AreEqual(3, layer[2][3]);

      res = layer.TryUpdateParams(pars, true, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(11, cursor);
      Assert.AreEqual(3, layer[0][0]);
      Assert.AreEqual(2, layer[0][1]);
      Assert.AreEqual(-2, layer[1][0]);
      Assert.AreEqual(4, layer[1][2]);
      Assert.AreEqual(6, layer[2][3]);

      res = layer.TryUpdateParams(pars, true, ref cursor);
      Assert.IsFalse(res);
    }

    [TestMethod]
    public void NeuralSparseLayer_UseBias_Calculate()
    {
      var layer = new SimpleSparseLayer(true);
      layer.Build();

      var input = new double[] { 1, 1, 1, 1 };
      var res = layer.Calculate(input);
      Assert.AreEqual(3,  res.Length);
      Assert.AreEqual(2,  res[0]);
      Assert.AreEqual(6,  res[1]);
      Assert.AreEqual(-3, res[2]);

      layer.TrySetParam(0,  2, false);
      layer.TrySetParam(2, -3, true);
      layer.TrySetParam(4,  1, false);
      layer.TrySetParam(7, -3, true);

      res = layer.Calculate(input);
      Assert.AreEqual(3,  res.Length);
      Assert.AreEqual(0,  res[0]);
      Assert.AreEqual(4,  res[1]);
      Assert.AreEqual(-6, res[2]);

      var cursor = 2;
      layer.TryUpdateParams(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 0 }, false, ref cursor);
      res = layer.Calculate(input);
      Assert.AreEqual(3,  res.Length);
      Assert.AreEqual(12, res[0]);
      Assert.AreEqual(21, res[1]);
      Assert.AreEqual(9,  res[2]);

      cursor = 2;
      layer.TryUpdateParams(new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, true, ref cursor);
      res = layer.Calculate(input);
      Assert.AreEqual(3,  res.Length);
      Assert.AreEqual(15, res[0]);
      Assert.AreEqual(24, res[1]);
      Assert.AreEqual(11, res[2]);
    }

    [TestMethod]
    public void NeuralFullLayer_UseBias_Calculate()
    {
      var layer = new SimpleFullLayer(true);
      layer.Build();

      var input = new double[] { 1, 1, 1, 1 };
      var res = layer.Calculate(input);
      Assert.AreEqual(3,  res.Length);
      Assert.AreEqual(2,  res[0]);
      Assert.AreEqual(6,  res[1]);
      Assert.AreEqual(-3, res[2]);

      layer.TrySetParam(0,   2, false);
      layer.TrySetParam(2,  -3, true);
      layer.TrySetParam(4,   1, false);
      layer.TrySetParam(7,  -3, true);
      layer.TrySetParam(11, -2, false);
      layer.TrySetParam(14, -1, true);

      res = layer.Calculate(input);
      Assert.AreEqual(3,  res.Length);
      Assert.AreEqual(-1, res[0]);
      Assert.AreEqual(3,  res[1]);
      Assert.AreEqual(-6, res[2]);

      var cursor = 2;
      layer.TryUpdateParams(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, -1, -2, -3, -4, -5, -6, -7 }, false, ref cursor);
      res = layer.Calculate(input);
      Assert.AreEqual(3,  res.Length);
      Assert.AreEqual(25, res[0]);
      Assert.AreEqual(14, res[1]);
      Assert.AreEqual(-25,  res[2]);

      cursor = 2;
      layer.TryUpdateParams(new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, true, ref cursor);
      res = layer.Calculate(input);
      Assert.AreEqual(3,  res.Length);
      Assert.AreEqual(30,  res[0]);
      Assert.AreEqual(19,  res[1]);
      Assert.AreEqual(-20, res[2]);
    }

    [TestMethod]
    public void NeuralLayer_Bench_Calculate()
    {
      var incount = 1000;
      var ncount = 10000;
      var layer = new NeuralLayer(incount);
      for (int i = 0; i < ncount; i++)
      {
        var n = layer.CreateNeuron<SparseNeuron>();
        for (int j = 0; j < incount; j++)
          n[j] = ((double)(i + j)) / incount;
      }

      layer.Build();

      var input = new double[incount];
      for (int i = 0; i < incount; i++)
        input[i] = i;

      var times = 100;
      var timer = new Stopwatch();
      timer.Start();
      for (int t = 0; t < times; t++)
        layer.Calculate(input);
      timer.Stop();

      Console.WriteLine("NeuralLayer_Bench_Calculate: input={0} neurons={1} elapsed={2}ms", incount, ncount, (int)(timer.Elapsed.TotalMilliseconds / times));
    }

    [TestMethod]
    public void NeuralLayer_Bench_BulkSetVSIndexSet()
    {
      var incount = 1000;
      var ncount = 1000;
      int pcount = incount * ncount;

      var layer = new NeuralLayer(incount);
      for (int i = 0; i < ncount; i++)
      {
        var n = layer.CreateNeuron<SparseNeuron>();
        for (int j = 0; j < incount; j++)
          n[j] = ((double)(i + j)) / incount;
      }

      layer.Build();

      var pars = new double[pcount];
      pars[1234] = 1.0D;
      pars[123404] = -1.0D;
      pars[200404] = -2.0D;
      pars[700404] = 3.0D;

      var times = 100;
      var timer = new Stopwatch();
      timer.Start();
      for (int i = 0; i < times; i++)
      {
        var cursor = 0;
        layer.TryUpdateParams(pars, true, ref cursor);
      }
      timer.Stop();

      Console.WriteLine("Layer Bulk set BM: input={0} neurons={1} elapsed={2}ms", incount, ncount, (int)(timer.Elapsed.TotalMilliseconds / times));

      timer.Reset();
      timer.Start();
      for (int i = 0; i < times; i++)
      {
        layer.TrySetParam(1234, 1.0D, true);
        layer.TrySetParam(123404, -1.0D, true);
        layer.TrySetParam(200404, -2.0D, true);
        layer.TrySetParam(700404, 3.0D, true);
      }
      timer.Stop();

      Console.WriteLine("Layer Index set BM: input={0} neurons={1} elapsed={2}ms", incount, ncount, Math.Round(timer.Elapsed.TotalMilliseconds / times, 2));
    }

    #endregion

    #region NeuralNetwork

    [TestMethod]
    public void NeuralNetwork_CreateHiddenLayer()
    {
      var net = new NeuralNetwork() { InputDim = 2 };
      var l = net.CreateLayer();

      Assert.IsNotNull(l);
    }

    [TestMethod]
    public void NeuralNetwork_Build()
    {
      var net = new SimpleNetwork<SparseNeuron>();
      net.Build();

      Assert.AreEqual(3, net.SubNodes.Length);

      Assert.AreEqual(1,  net[0][0].ParamCount);
      Assert.AreEqual(-1, net[0][0][0]);
      Assert.AreEqual(0,  net[0][0][1]);
      Assert.AreEqual(0,  net[0][0][2]);
      Assert.AreEqual(2,  net[0][1].ParamCount);
      Assert.AreEqual(0,  net[0][1][0]);
      Assert.AreEqual(2,  net[0][1][1]);
      Assert.AreEqual(3,  net[0][1][2]);
      Assert.AreEqual(1,  net[0][2].ParamCount);
      Assert.AreEqual(0,  net[0][2][0]);
      Assert.AreEqual(0,  net[0][2][1]);
      Assert.AreEqual(-2, net[0][2][2]);

      Assert.AreEqual(2,  net[1][0].ParamCount);
      Assert.AreEqual(1,  net[1][0][0]);
      Assert.AreEqual(-1, net[1][0][1]);
      Assert.AreEqual(0,  net[1][0][2]);
      Assert.AreEqual(1,  net[1][1].ParamCount);
      Assert.AreEqual(0,  net[1][1][0]);
      Assert.AreEqual(0,  net[1][1][1]);
      Assert.AreEqual(3,  net[1][1][2]);

      Assert.AreEqual(2,  net[2][0].ParamCount);
      Assert.AreEqual(1,  net[2][0][0]);
      Assert.AreEqual(-1, net[2][0][1]);
    }

    [TestMethod]
    public void NeuralNetwork_Calculate()
    {
      var net = new SimpleNetwork<SparseNeuron>();
      net.Build();

      var res = net.Calculate(new double[] { 1, 2, 3 });

      Assert.AreEqual(4, res[0]);
    }

    [TestMethod]
    public void NeuralNetwork_TryGetParam()
    {
      var net = new SimpleNetwork<SparseNeuron>();
      net.Build();

      double par1;
      double par2;
      double par3;
      double par4;
      double par5;
      double par6;
      double par7;
      double par8;
      double par9;
      double par10;
      var res1 = net.TryGetParam(0, out par1);
      var res2 = net.TryGetParam(1, out par2);
      var res3 = net.TryGetParam(2, out par3);
      var res4 = net.TryGetParam(3, out par4);
      var res5 = net.TryGetParam(4, out par5);
      var res6 = net.TryGetParam(5, out par6);
      var res7 = net.TryGetParam(6, out par7);
      var res8 = net.TryGetParam(7, out par8);
      var res9 = net.TryGetParam(8, out par9);
      var res10 = net.TryGetParam(9, out par10);

      Assert.IsTrue(res1);
      Assert.IsTrue(res2);
      Assert.IsTrue(res3);
      Assert.IsTrue(res4);
      Assert.IsTrue(res5);
      Assert.IsTrue(res6);
      Assert.IsTrue(res7);
      Assert.IsTrue(res8);
      Assert.IsTrue(res9);
      Assert.IsFalse(res10);
      Assert.AreEqual(-1, par1);
      Assert.AreEqual(2, par2);
      Assert.AreEqual(3, par3);
      Assert.AreEqual(-2, par4);
      Assert.AreEqual(1, par5);
      Assert.AreEqual(-1, par6);
      Assert.AreEqual(3, par7);
      Assert.AreEqual(1, par8);
      Assert.AreEqual(-1, par9);
    }

    [TestMethod]
    public void NeuralNetwork_TrySetParam()
    {
      var net = new SimpleNetwork<SparseNeuron>();
      net.Build();

      var res1 = net.TrySetParam(0,  1, false);
      var res2 = net.TrySetParam(1,  1, true);
      var res3 = net.TrySetParam(2, -2, false);
      var res4 = net.TrySetParam(3, -4, true);
      var res5 = net.TrySetParam(4,  1, false);
      var res6 = net.TrySetParam(5,  2, true);
      var res7 = net.TrySetParam(6, -3, false);
      var res8 = net.TrySetParam(7, -1, true);
      var res9 = net.TrySetParam(8,  2, false);
      var res10 = net.TrySetParam(9, 4, false);

      Assert.IsTrue(res1);
      Assert.IsTrue(res2);
      Assert.IsTrue(res3);
      Assert.IsTrue(res4);
      Assert.IsTrue(res5);
      Assert.IsTrue(res6);
      Assert.IsTrue(res7);
      Assert.IsTrue(res8);
      Assert.IsTrue(res9);
      Assert.IsFalse(res10);
      Assert.AreEqual(1,  net[0][0][0]);
      Assert.AreEqual(0,  net[0][0][1]);
      Assert.AreEqual(0,  net[0][0][2]);
      Assert.AreEqual(0,  net[0][1][0]);
      Assert.AreEqual(3,  net[0][1][1]);
      Assert.AreEqual(-2, net[0][1][2]);
      Assert.AreEqual(0,  net[0][2][0]);
      Assert.AreEqual(0,  net[0][2][1]);
      Assert.AreEqual(-6, net[0][2][2]);
      Assert.AreEqual(1,  net[1][0][0]);
      Assert.AreEqual(1,  net[1][0][1]);
      Assert.AreEqual(0,  net[1][0][2]);
      Assert.AreEqual(0,  net[1][1][0]);
      Assert.AreEqual(0,  net[1][1][1]);
      Assert.AreEqual(-3, net[1][1][2]);
      Assert.AreEqual(0,  net[2][0][0]);
      Assert.AreEqual(2,  net[2][0][1]);
    }

    [TestMethod]
    public void NeuralNetwork_TryUpdateParams()
    {
      var net = new SimpleNetwork<SparseNeuron>();
      net.Build();

      var pars = new double[] { 1, 2, 3, -4, -1, 3, 1, -1, 2, 5, 3, 5 };
      var cursor = 1;

      var res = net.TryUpdateParams(pars, false, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(10, cursor);
      Assert.AreEqual(2,  net[0][0][0]);
      Assert.AreEqual(0,  net[0][0][1]);
      Assert.AreEqual(0,  net[0][0][2]);
      Assert.AreEqual(0,  net[0][1][0]);
      Assert.AreEqual(3,  net[0][1][1]);
      Assert.AreEqual(-4, net[0][1][2]);
      Assert.AreEqual(0,  net[0][2][0]);
      Assert.AreEqual(0,  net[0][2][1]);
      Assert.AreEqual(-1, net[0][2][2]);
      Assert.AreEqual(3,  net[1][0][0]);
      Assert.AreEqual(1,  net[1][0][1]);
      Assert.AreEqual(0,  net[1][0][2]);
      Assert.AreEqual(0,  net[1][1][0]);
      Assert.AreEqual(0,  net[1][1][1]);
      Assert.AreEqual(-1, net[1][1][2]);
      Assert.AreEqual(2,  net[2][0][0]);
      Assert.AreEqual(5,  net[2][0][1]);

      cursor = 0;
      res = net.TryUpdateParams(pars, true, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(9, cursor);
      Assert.AreEqual(3,  net[0][0][0]);
      Assert.AreEqual(0,  net[0][0][1]);
      Assert.AreEqual(0,  net[0][0][2]);
      Assert.AreEqual(0,  net[0][1][0]);
      Assert.AreEqual(5,  net[0][1][1]);
      Assert.AreEqual(-1, net[0][1][2]);
      Assert.AreEqual(0,  net[0][2][0]);
      Assert.AreEqual(0,  net[0][2][1]);
      Assert.AreEqual(-5, net[0][2][2]);
      Assert.AreEqual(2,  net[1][0][0]);
      Assert.AreEqual(4,  net[1][0][1]);
      Assert.AreEqual(0,  net[1][0][2]);
      Assert.AreEqual(0,  net[1][1][0]);
      Assert.AreEqual(0,  net[1][1][1]);
      Assert.AreEqual(0,  net[1][1][2]);
      Assert.AreEqual(1,  net[2][0][0]);
      Assert.AreEqual(7,  net[2][0][1]);

      res = net.TryUpdateParams(pars, true, ref cursor);
      Assert.IsFalse(res);
    }

    [TestMethod]
    public void NeuralNetwork_Bench_Calculate()
    {
      var ncount = 1000;
      var lcount = 10;
      var net = new LargeNetwork(lcount, ncount);
      net.Build();

      var input = new double[ncount];
      for (int i = 0; i < ncount; i++)
        input[i] = i;

      var times = 50;
      var timer = new Stopwatch();
      timer.Start();
      for (int t = 0; t < times; t++)
        net.Calculate(input);
      timer.Stop();

      Console.WriteLine("NeuralNetwork_Bench_Calculate: layers={0} neurons={1} elapsed={2}ms", lcount, ncount, (int)(timer.Elapsed.TotalMilliseconds / times));
    }

    [TestMethod]
    public void NeuralNetwork_Bench_BulkSetVSIndexSet()
    {
      var ncount = 1000;
      var lcount = 10;
      var pcount = ncount * ncount * lcount;
      var net = new LargeNetwork(lcount, ncount);
      net.Build();

      var pars = new double[pcount];
      pars[1234] = 1.0D;
      pars[123404] = -1.0D;
      pars[1200404] = -2.0D;
      pars[9700404] = 3.0D;

      var times = 30;
      var timer = new Stopwatch();
      timer.Start();
      for (int i = 0; i < times; i++)
      {
        var cursor = 0;
        net.TryUpdateParams(pars, true, ref cursor);
      }
      timer.Stop();

      Console.WriteLine("Network Bulk set BM: layers={0} neurons={1} elapsed={2}ms", lcount, ncount, (int)(timer.Elapsed.TotalMilliseconds / times));

      timer.Reset();
      timer.Start();
      for (int i = 0; i < times; i++)
      {
        net.TrySetParam(1234, 1.0D, true);
        net.TrySetParam(123404, -1.0D, true);
        net.TrySetParam(200404, -2.0D, true);
        net.TrySetParam(700404, 3.0D, true);
      }
      timer.Stop();

      Console.WriteLine("Network Index set BM: layers={0} neurons={1} elapsed={2}ms", lcount, ncount, Math.Round(timer.Elapsed.TotalMilliseconds / times, 2));
    }


    #endregion

    //[TestMethod]
    //public void ___Test()
    //{
    //  var len = 40000;
    //  var n1 = new FullNeuron(len, false);
    //  var n2 = new SparseNeuron(len, false);
    //  for (int i=0; i<len; i++)
    //  {
    //    n1[i] = i;
    //    n2[i] = i;
    //  }
    //  n1.Build();
    //  n2.Build();

    //  var times = 1;
    //  var timer = new Stopwatch();

    //  timer.Restart();
    //  for (int t=0; t<times; t++)
    //  for (int i=0; i<len; i++)
    //  {
    //    double value;
    //    n2.TryGetParam(i, out value);
    //    n2.TrySetParam(i, 234, false);
    //  }
    //  timer.Stop();
    //  Console.WriteLine("sparse neuron: {0}", timer.Elapsed.TotalMilliseconds/times);

    //  timer.Restart();
    //  for (int t=0; t<times; t++)
    //  for (int i=0; i<len; i++)
    //  {
    //    double value;
    //    n1.TryGetParam(i, out value);
    //    n1.TrySetParam(i, 234, false);
    //  }
    //  timer.Stop();
    //  Console.WriteLine("full neuron: {0}", timer.Elapsed.TotalMilliseconds/times);

    //}
  }
}
