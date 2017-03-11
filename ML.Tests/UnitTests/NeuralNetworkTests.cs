//using System;
//using System.Diagnostics;
//using Microsoft.VisualStudio.TestTools.UnitTesting;
//using ML.Core;
//using ML.Core.ComputingNetworks;
//using ML.NeuralMethods.Model;

//namespace ML.Tests.UnitTests
//{
//  [TestClass]
//  public class NeuralNetworkTests : TestBase
//  {
//    public const double EPS = 0.0000001D;

//    #region Inner

//    public class SimpleLayer : NeuralLayer
//    {
//      public SimpleLayer() : base(4)
//      {
//        var n1 = this.CreateNeuron<Neuron>();
//        n1[0] = 1;
//        n1[1] = -1;
//        n1.Bias = 2;

//        var n2 = this.CreateNeuron<Neuron>();
//        n2[0] = 2;
//        n2[2] = 3;
//        n2.Bias = 1;

//        var n3 = this.CreateNeuron<Neuron>();
//        n3[3] = -2;
//        n3.Bias = -1;
//      }
//    }

//    public class SimpleNetwork : NeuralNetwork
//    {
//      public SimpleNetwork() : base(3)
//      {
//        var l1 = this.CreateLayer<NeuralLayer>();
//        var n11 = l1.CreateNeuron<Neuron>();
//        n11[0] = -1;
//        var n12 = l1.CreateNeuron<Neuron>();
//        n12[1] = 2;
//        n12[2] = 3;
//        var n13 = l1.CreateNeuron<Neuron>();
//        n13[2] = -2;

//        var l2 = this.CreateLayer<NeuralLayer>();
//        var n21 = l2.CreateNeuron<Neuron>();
//        n21[0] = 1;
//        n21[1] = -1;
//        var n22 = l2.CreateNeuron<Neuron>();
//        n22[2] = 3;

//        var output = this.CreateLayer<NeuralLayer>();
//        var on = output.CreateNeuron<Neuron>();
//        on[0] = 1;
//        on[1] = -1;
//      }
//    }

//    public class LargeNetwork : NeuralNetwork
//    {
//      public LargeNetwork(int lcount, int ncount) : base(ncount)
//      {
//        for (int i = 0; i < lcount; i++)
//        {
//          var layer = this.CreateLayer<NeuralLayer>();
//          for (int j = 0; j < ncount; j++)
//          {
//            var n = layer.CreateNeuron<Neuron>();
//            for (int k = 0; k < ncount; k++)
//              n[k] = ((double)(i + j + k)) / (ncount * ncount);
//          }
//        }

//        var output = this.CreateLayer<NeuralLayer>();
//        var on = output.CreateNeuron<Neuron>();
//        for (int i = 0; i < ncount; i++)
//          on[i] = (i % 2 == 0) ? 1 : -1;
//      }
//    }

//    #endregion

//    [ClassInitialize]
//    public static void ClassInit(TestContext context)
//    {
//      BaseClassInit(context);
//    }

//    #region Neuron

//    [TestMethod]
//    public void Neuron_Build()
//    {
//      var n = new Neuron(4);
//      n[0] = 0.1D;
//      n[2] = -0.2D;
//      n[3] = 0.3D;
//      n.Bias = 0.5D;
//      n.Build();

//      Assert.AreEqual(5, n.ParamCount);
//    }

//    [TestMethod]
//    [ExpectedException(typeof(MLException))]
//    public void Neuron_Calculate_WrongInput()
//    {
//      var n = new Neuron(4);
//      n[0] = 0.1D;
//      n[2] = -0.2D;
//      n[3] = 0.3D;
//      n.Build();
//      var input = new double[] { 1, 2, 3, 4, 5 };

//      n.Calculate(input);
//    }

//    [TestMethod]
//    public void Neuron_Calculate()
//    {
//      var n = new Neuron(4);
//      n[0] = 0.1D;
//      n[2] = -0.2D;
//      n[3] = 0.3D;
//      n.Bias = 0.5;
//      n.Build();
//      var input = new double[] { 1, 2, 3, 4 };

//      var result = n.Calculate(input);

//      Assert.AreEqual(1.2D, result, EPS);
//      Assert.AreEqual(1.2D, n.Value,      EPS);
//      Assert.AreEqual(1.2D, n.NetValue,   EPS);
//      Assert.AreEqual(1.0D, n.Derivative, EPS);
//    }

//    [TestMethod]
//    public void Neuron_Calculate_ActivationFunction()
//    {
//      var n = new Neuron(4);
//      n[0] = 0.1D;
//      n[2] = -0.2D;
//      n[3] = 0.3D;
//      n.ActivationFunction = Registry.ActivationFunctions.Exp;
//      n.Build();
//      var input = new double[] { 1, 2, 3, 4 };

//      var result = n.Calculate(input);

//      Assert.AreEqual(Math.Exp(0.7D), result,       EPS);
//      Assert.AreEqual(Math.Exp(0.7D), n.Value,      EPS);
//      Assert.AreEqual(0.7D,           n.NetValue,   EPS);
//      Assert.AreEqual(Math.Exp(0.7D), n.Derivative, EPS);
//    }

//    [TestMethod]
//    public void Neuron_Indexer()
//    {
//      var n = new Neuron(4);
//      n[0] = 0.1D;
//      n[2] = -0.2D;
//      n[3] = 0.3D;
//      n.Bias = 0.5D;
//      n.Build();

//      Assert.AreEqual(0.1D,  n[0]);
//      Assert.AreEqual(0,     n[1]);
//      Assert.AreEqual(-0.2D, n[2]);
//      Assert.AreEqual(0.3D,  n[3]);
//      Assert.AreEqual(0.5D,  n.Bias);
//    }

//    [TestMethod]
//    public void Neuron_TryGetParam()
//    {
//      var n = new Neuron(4);
//      n[0] = 0.1D;
//      n[2] = -0.2D;
//      n[3] = 0.3D;
//      n.Bias = 0.5D;
//      n.Build();

//      double par1;
//      double par2;
//      double par3;
//      double par4;
//      double par5;
//      double par6;
//      var res1 = n.TryGetParam(0, out par1);
//      var res2 = n.TryGetParam(1, out par2);
//      var res3 = n.TryGetParam(2, out par3);
//      var res4 = n.TryGetParam(3, out par4);
//      var res5 = n.TryGetParam(4, out par5);
//      var res6 = n.TryGetParam(5, out par6);

//      Assert.IsTrue(res1);
//      Assert.IsTrue(res2);
//      Assert.IsTrue(res3);
//      Assert.IsTrue(res4);
//      Assert.IsTrue(res5);
//      Assert.IsFalse(res6);
//      Assert.AreEqual(0.1D, par1);
//      Assert.AreEqual(0, par2);
//      Assert.AreEqual(-0.2D, par3);
//      Assert.AreEqual(0.3D, par4);
//      Assert.AreEqual(0.5D, par5);
//    }

//    [TestMethod]
//    public void Neuron_rySetParam()
//    {
//      var n = new Neuron(4);
//      n[0] = 0.1D;
//      n[2] = -0.2D;
//      n[3] = 0.3D;
//      n.Bias = 0.5D;
//      n.Build();

//      var res1 = n.TrySetParam(0, -1, false);
//      var res2 = n.TrySetParam(1,  1, true);
//      var res3 = n.TrySetParam(2,  2, false);
//      var res4 = n.TrySetParam(3,  3, true);
//      var res5 = n.TrySetParam(4,  4, true);
//      var res6 = n.TrySetParam(5,  5, true);

//      Assert.IsTrue(res1);
//      Assert.IsTrue(res2);
//      Assert.IsTrue(res3);
//      Assert.IsTrue(res4);
//      Assert.IsTrue(res5);
//      Assert.IsFalse(res6);
//      Assert.AreEqual(-1,   n[0]);
//      Assert.AreEqual(1,    n[1]);
//      Assert.AreEqual(2,    n[2]);
//      Assert.AreEqual(3.3D, n[3]);
//      Assert.AreEqual(4.5D, n.Bias);
//    }

//    [TestMethod]
//    public void Neuron_TryUpdateParams()
//    {
//      var n = new Neuron(4);
//      n[0] = 0.1D;
//      n[2] = -0.2D;
//      n[3] = 0.3D;
//      n.Bias = 0.5D;
//      n.Build();

//      var pars = new double[] { 1, 2, -1, 3, 1, 2, -3, -1, 1, 7, -1, 1 };
//      var cursor = 1;

//      var res = n.TryUpdateParams(pars, false, ref cursor);
//      Assert.IsTrue(res);
//      Assert.AreEqual(6, cursor);
//      Assert.AreEqual(2,  n[0]);
//      Assert.AreEqual(-1, n[1]);
//      Assert.AreEqual(3,  n[2]);
//      Assert.AreEqual(1,  n[3]);
//      Assert.AreEqual(2,  n.Bias);

//      res = n.TryUpdateParams(pars, true, ref cursor);
//      Assert.IsTrue(res);
//      Assert.AreEqual(11, cursor);
//      Assert.AreEqual(-1, n[0]);
//      Assert.AreEqual(-2, n[1]);
//      Assert.AreEqual(4,  n[2]);
//      Assert.AreEqual(8,  n[3]);
//      Assert.AreEqual(1,  n.Bias);

//      res = n.TryUpdateParams(pars, false, ref cursor);
//      Assert.IsFalse(res);
//    }

//    [TestMethod]
//    public void Neuron_ComplexCalculation()
//    {
//      var cinput = 11;
//      var n = new Neuron(cinput);
//      var input = new double[cinput];
//      for (int i = 0; i < cinput; i++)
//      {
//        if (i % 2 == 0) n[i] = i;
//        input[i] = 1;
//      }
//      n.Build();

//      var result = n.Calculate(input);
//      Assert.AreEqual(30, result);

//      var success = n.TrySetParam(0, 1, false);
//      Assert.IsTrue(success);
//      success = n.TrySetParam(1, -2, true);
//      Assert.IsTrue(success);

//      result = n.Calculate(input);
//      Assert.AreEqual(29, result);

//      var pars = new double[n.ParamCount];
//      for (int i = 0; i < n.ParamCount; i++)
//        pars[i] = 1;
//      var cursor = 0;
//      n.TryUpdateParams(pars, true, ref cursor);

//      result = n.Calculate(input);
//      Assert.AreEqual(41, result);

//      cursor = 0;
//      n.TryUpdateParams(pars, false, ref cursor);

//      result = n.Calculate(input);
//      Assert.AreEqual(12, result);
//    }

//    [TestMethod]
//    public void Neuron_Bench_Calculate()
//    {
//      int cinput = 1000000;
//      var n = new Neuron(cinput);
//      var input = new double[cinput];
//      for (int i = 0; i < cinput; i++)
//      {
//        if (i % 9 != 0) n[i] = ((double)(i % 1234)) / 1000;
//        input[i] = ((double)i) / 10000;
//      }
//      n.Build();

//      var times = 200;
//      var timer = new Stopwatch();
//      timer.Start();
//      for (int i = 0; i < times; i++)
//        n.Calculate(input);
//      timer.Stop();

//      Console.WriteLine("Neuron Calculate BM: cinput={0} elapsed={1}ms", cinput, (int)(timer.Elapsed.TotalMilliseconds / times));
//    }

//    [TestMethod]
//    public void Neuron_Bench_TryUpdateParams_Calculate()
//    {
//      int cinput = 1000000;
//      var n = new Neuron(cinput);
//      for (int i = 0; i < cinput; i++)
//      {
//        if (i % 9 != 0) n[i] = ((double)(i % 1234)) / 1000;
//      }
//      n.Build();
//      var pars = new double[n.ParamCount];
//      for (int i = 0; i < pars.Length; i++)
//        pars[i] = ((double)i) / 10000;

//      var times = 200;
//      var timer = new Stopwatch();
//      timer.Start();
//      for (int i = 0; i < times; i++)
//      {
//        var cursor = 0;
//        var res = n.TryUpdateParams(pars, true, ref cursor);
//        if (!res) throw new MLCorruptedIndexException();
//      }
//      timer.Stop();

//      Console.WriteLine("Neuron update params BM: cinput={0} elapsed={1}ms", cinput, (int)(timer.Elapsed.TotalMilliseconds / times));
//    }

//    [TestMethod]
//    public void Neuron_Bench_BulkSetVSIndexSet()
//    {
//      int cinput = 1000000;
//      var n = new Neuron(cinput);
//      for (int i = 0; i < cinput; i++)
//      {
//        if (i % 9 != 0) n[i] = ((double)(i % 1234)) / 1000;
//      }
//      n.Build();

//      var pars = new double[n.ParamCount];
//      pars[1234] = 1.0D;
//      pars[123404] = -1.0D;
//      pars[200404] = -2.0D;
//      pars[700404] = 3.0D;

//      var times = 200;
//      var timer = new Stopwatch();
//      timer.Start();
//      for (int i = 0; i < times; i++)
//      {
//        var cursor = 0;
//        n.TryUpdateParams(pars, true, ref cursor);
//      }
//      timer.Stop();

//      Console.WriteLine("Neuron Bulk set BM: cinput={0} elapsed={1}ms", cinput, (int)(timer.Elapsed.TotalMilliseconds / times));

//      timer.Reset();
//      timer.Start();
//      for (int i = 0; i < times; i++)
//      {
//        n.TrySetParam(1234, 1.0D, true);
//        n.TrySetParam(123404, -1.0D, true);
//        n.TrySetParam(200404, -2.0D, true);
//        n.TrySetParam(700404, 3.0D, true);
//      }
//      timer.Stop();

//      Console.WriteLine("Neuron Index set BM: cinput={0} elapsed={1}ms", cinput, (int)(timer.Elapsed.TotalMilliseconds / times));
//    }

//    #endregion

//    #region NeuralLayer

//    [TestMethod]
//    public void NeuralLayer_CreateNeuron()
//    {
//      var layer = new NeuralLayer(12);
//      var n1 = layer.CreateNeuron<Neuron>();

//      Assert.IsNotNull(n1);
//    }

//    [TestMethod]
//    public void NeuralLayer_Build()
//    {
//      var layer = new SimpleLayer();
//      layer.Build();

//      Assert.AreEqual(3, layer.SubNodes.Length);
//      Assert.AreEqual(3, layer[0].ParamCount);
//      Assert.AreEqual(3, layer[1].ParamCount);
//      Assert.AreEqual(2, layer[2].ParamCount);
//    }

//    [TestMethod]
//    public void NeuralLayer_Calculate()
//    {
//      var layer = new SimpleLayer();
//      layer.Build();

//      var res1 = layer.Calculate(new double[] { 1, 1, 1, 1 });
//      Assert.AreEqual( 3, res1.Length);
//      Assert.AreEqual( 2, res1[0]);
//      Assert.AreEqual( 6, res1[1]);
//      Assert.AreEqual(-3, res1[2]);
//    }

//    [TestMethod]
//    public void NeuralLayer_TryGetParam()
//    {
//      var layer = new SimpleLayer();
//      layer.Build();

//      double par1;
//      double par2;
//      double par3;
//      double par4;
//      double par5;
//      double par6;
//      double par7;
//      var res1 = layer.TryGetParam(0, out par1);
//      var res2 = layer.TryGetParam(1, out par2);
//      var res3 = layer.TryGetParam(3, out par3);
//      var res4 = layer.TryGetParam(4, out par4);
//      var res5 = layer.TryGetParam(6, out par5);
//      var res6 = layer.TryGetParam(7, out par6);
//      var res7 = layer.TryGetParam(8, out par7);

//      Assert.IsTrue(res1);
//      Assert.AreEqual(1, par1);
//      Assert.IsTrue(res2);
//      Assert.AreEqual(-1, par2);
//      Assert.IsTrue(res3);
//      Assert.AreEqual(2, par3);
//      Assert.IsTrue(res4);
//      Assert.AreEqual(3, par4);
//      Assert.IsTrue(res5);
//      Assert.AreEqual(-2, par5);
//      Assert.IsTrue(res6);
//      Assert.AreEqual(-1, par6);
//      Assert.IsFalse(res7);
//      Assert.AreEqual(0, par7);
//    }

//    [TestMethod]
//    public void NeuralLayer_TrySetParam()
//    {
//      var layer = new SimpleLayer();
//      layer.Build();

//      var res1 = layer.TrySetParam(0,  1, false);
//      var res2 = layer.TrySetParam(1,  1, true);
//      var res3 = layer.TrySetParam(3, -2, false);
//      var res4 = layer.TrySetParam(4, -4, true);
//      var res5 = layer.TrySetParam(6,  4, false);
//      var res6 = layer.TrySetParam(7,  5, false);
//      var res7 = layer.TrySetParam(8,  4, false);

//      Assert.IsTrue(res1);
//      Assert.AreEqual(1, layer[0][0]);
//      Assert.IsTrue(res2);
//      Assert.AreEqual(0, layer[0][1]);
//      Assert.IsTrue(res3);
//      Assert.AreEqual(-2, layer[1][0]);
//      Assert.IsTrue(res4);
//      Assert.AreEqual(-1, layer[1][2]);
//      Assert.IsTrue(res5);
//      Assert.AreEqual(4, layer[2][3]);
//      Assert.IsTrue(res6);
//      Assert.AreEqual(5, layer[2].Bias);
//      Assert.IsFalse(res7);
//    }

//    [TestMethod]
//    public void NeuralLayer_TryUpdateParams()
//    {
//      var layer = new SimpleLayer();
//      layer.Build();

//      var pars = new double[] { 1, 2, 3, 1, -4, -1, 2, 3, 2, 1 };
//      var cursor = 1;

//      var res = layer.TryUpdateParams(pars, false, ref cursor);
//      Assert.IsTrue(res);
//      Assert.AreEqual(9, cursor);
//      Assert.AreEqual( 2, layer[0][0]);
//      Assert.AreEqual( 3, layer[0][1]);
//      Assert.AreEqual( 1, layer[0].Bias);
//      Assert.AreEqual(-4, layer[1][0]);
//      Assert.AreEqual(-1, layer[1][2]);
//      Assert.AreEqual( 2, layer[1].Bias);
//      Assert.AreEqual( 3, layer[2][3]);
//      Assert.AreEqual( 2, layer[2].Bias);

//      cursor=1;
//      res = layer.TryUpdateParams(pars, true, ref cursor);
//      Assert.IsTrue(res);
//      Assert.AreEqual( 9, cursor);
//      Assert.AreEqual( 4, layer[0][0]);
//      Assert.AreEqual( 6, layer[0][1]);
//      Assert.AreEqual( 2, layer[0].Bias);
//      Assert.AreEqual(-8, layer[1][0]);
//      Assert.AreEqual(-2, layer[1][2]);
//      Assert.AreEqual( 4, layer[1].Bias);
//      Assert.AreEqual( 6, layer[2][3]);
//      Assert.AreEqual( 4, layer[2].Bias);

//      res = layer.TryUpdateParams(pars, true, ref cursor);
//      Assert.IsFalse(res);
//    }

//    [TestMethod]
//    public void NeuralSparseLayer_UseBias_Calculate()
//    {
//      var layer = new SimpleLayer();
//      layer.Build();

//      var input = new double[] { 1, 1, 1, 1 };
//      var res = layer.Calculate(input);
//      Assert.AreEqual(3,  res.Length);
//      Assert.AreEqual(2,  res[0]);
//      Assert.AreEqual(6,  res[1]);
//      Assert.AreEqual(-3, res[2]);

//      layer.TrySetParam(0,  2, false);
//      layer.TrySetParam(2, -3, true);
//      layer.TrySetParam(4,  1, false);
//      layer.TrySetParam(7, -3, true);

//      res = layer.Calculate(input);
//      Assert.AreEqual(3,  res.Length);
//      Assert.AreEqual(0,  res[0]);
//      Assert.AreEqual(4,  res[1]);
//      Assert.AreEqual(-6, res[2]);

//      var cursor = 2;
//      layer.TryUpdateParams(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 0 }, false, ref cursor);
//      res = layer.Calculate(input);
//      Assert.AreEqual(3,  res.Length);
//      Assert.AreEqual(12, res[0]);
//      Assert.AreEqual(21, res[1]);
//      Assert.AreEqual(9,  res[2]);

//      cursor = 2;
//      layer.TryUpdateParams(new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, true, ref cursor);
//      res = layer.Calculate(input);
//      Assert.AreEqual(3,  res.Length);
//      Assert.AreEqual(15, res[0]);
//      Assert.AreEqual(24, res[1]);
//      Assert.AreEqual(11, res[2]);
//    }

//    [TestMethod]
//    public void NeuralFullLayer_UseBias_Calculate()
//    {
//      var layer = new SimpleLayer();
//      layer.Build();

//      var input = new double[] { 1, 1, 1, 1 };
//      var res = layer.Calculate(input);
//      Assert.AreEqual(3,  res.Length);
//      Assert.AreEqual(2,  res[0]);
//      Assert.AreEqual(6,  res[1]);
//      Assert.AreEqual(-3, res[2]);

//      layer.TrySetParam(0,   2, false);
//      layer.TrySetParam(2,  -3, true);
//      layer.TrySetParam(4,   1, false);
//      layer.TrySetParam(7,  -3, true);
//      layer.TrySetParam(11, -2, false);
//      layer.TrySetParam(14, -1, true);

//      res = layer.Calculate(input);
//      Assert.AreEqual(3,  res.Length);
//      Assert.AreEqual(-1, res[0]);
//      Assert.AreEqual(3,  res[1]);
//      Assert.AreEqual(-6, res[2]);

//      var cursor = 2;
//      layer.TryUpdateParams(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, -1, -2, -3, -4, -5, -6, -7 }, false, ref cursor);
//      res = layer.Calculate(input);
//      Assert.AreEqual(3,  res.Length);
//      Assert.AreEqual(25, res[0]);
//      Assert.AreEqual(14, res[1]);
//      Assert.AreEqual(-25,  res[2]);

//      cursor = 2;
//      layer.TryUpdateParams(new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, true, ref cursor);
//      res = layer.Calculate(input);
//      Assert.AreEqual(  3, res.Length);
//      Assert.AreEqual( 30, res[0]);
//      Assert.AreEqual( 19, res[1]);
//      Assert.AreEqual(-20, res[2]);
//    }

//    [TestMethod]
//    public void NeuralLayer_AddNeuron()
//    {
//      var layer = new SimpleLayer();
//      var neuron = new Neuron(layer.InputDim);

//      layer.AddNeuron(neuron);

//      Assert.AreEqual(neuron, layer[layer.NeuronCount-1]);
//    }

//    [TestMethod]
//    public void NeuralLayer_Bench_Calculate()
//    {
//      var incount = 1000;
//      var ncount = 10000;
//      var layer = new NeuralLayer(incount);
//      for (int i = 0; i < ncount; i++)
//      {
//        var n = layer.CreateNeuron<Neuron>();
//        for (int j = 0; j < incount; j++)
//          n[j] = ((double)(i + j)) / incount;
//      }

//      layer.Build();

//      var input = new double[incount];
//      for (int i = 0; i < incount; i++)
//        input[i] = i;

//      var times = 100;
//      var timer = new Stopwatch();
//      timer.Start();
//      for (int t = 0; t < times; t++)
//        layer.Calculate(input);
//      timer.Stop();

//      Console.WriteLine("NeuralLayer_Bench_Calculate: input={0} neurons={1} elapsed={2}ms", incount, ncount, (int)(timer.Elapsed.TotalMilliseconds / times));
//    }

//    [TestMethod]
//    public void NeuralLayer_Bench_BulkSetVSIndexSet()
//    {
//      var incount = 1000;
//      var ncount = 1000;
//      int pcount = incount * ncount;

//      var layer = new NeuralLayer(incount);
//      for (int i = 0; i < ncount; i++)
//      {
//        var n = layer.CreateNeuron<Neuron>();
//        for (int j = 0; j < incount; j++)
//          n[j] = ((double)(i + j)) / incount;
//      }

//      layer.Build();

//      var pars = new double[pcount];
//      pars[1234] = 1.0D;
//      pars[123404] = -1.0D;
//      pars[200404] = -2.0D;
//      pars[700404] = 3.0D;

//      var times = 100;
//      var timer = new Stopwatch();
//      timer.Start();
//      for (int i = 0; i < times; i++)
//      {
//        var cursor = 0;
//        layer.TryUpdateParams(pars, true, ref cursor);
//      }
//      timer.Stop();

//      Console.WriteLine("Layer Bulk set BM: input={0} neurons={1} elapsed={2}ms", incount, ncount, (int)(timer.Elapsed.TotalMilliseconds / times));

//      timer.Reset();
//      timer.Start();
//      for (int i = 0; i < times; i++)
//      {
//        layer.TrySetParam(1234, 1.0D, true);
//        layer.TrySetParam(123404, -1.0D, true);
//        layer.TrySetParam(200404, -2.0D, true);
//        layer.TrySetParam(700404, 3.0D, true);
//      }
//      timer.Stop();

//      Console.WriteLine("Layer Index set BM: input={0} neurons={1} elapsed={2}ms", incount, ncount, Math.Round(timer.Elapsed.TotalMilliseconds / times, 2));
//    }

//    #endregion

//    #region NeuralNetwork

//    [TestMethod]
//    public void NeuralNetwork_CreateHiddenLayer()
//    {
//      var net = new NeuralNetwork(2);
//      var l = net.CreateLayer<NeuralLayer>();

//      Assert.IsNotNull(l);
//    }

//    [TestMethod]
//    public void NeuralNetwork_Build()
//    {
//      var net = new SimpleNetwork();
//      net.Build();

//      Assert.AreEqual(3, net.SubNodes.Length);

//      Assert.AreEqual(2,  net[0][0].ParamCount);
//      Assert.AreEqual(-1, net[0][0][0]);
//      Assert.AreEqual(0,  net[0][0][1]);
//      Assert.AreEqual(0,  net[0][0][2]);
//      Assert.AreEqual(3,  net[0][1].ParamCount);
//      Assert.AreEqual(0,  net[0][1][0]);
//      Assert.AreEqual(2,  net[0][1][1]);
//      Assert.AreEqual(3,  net[0][1][2]);
//      Assert.AreEqual(2,  net[0][2].ParamCount);
//      Assert.AreEqual(0,  net[0][2][0]);
//      Assert.AreEqual(0,  net[0][2][1]);
//      Assert.AreEqual(-2, net[0][2][2]);

//      Assert.AreEqual(3,  net[1][0].ParamCount);
//      Assert.AreEqual(1,  net[1][0][0]);
//      Assert.AreEqual(-1, net[1][0][1]);
//      Assert.AreEqual(0,  net[1][0][2]);
//      Assert.AreEqual(2,  net[1][1].ParamCount);
//      Assert.AreEqual(0,  net[1][1][0]);
//      Assert.AreEqual(0,  net[1][1][1]);
//      Assert.AreEqual(3,  net[1][1][2]);

//      Assert.AreEqual(3,  net[2][0].ParamCount);
//      Assert.AreEqual(1,  net[2][0][0]);
//      Assert.AreEqual(-1, net[2][0][1]);
//    }

//    [TestMethod]
//    public void NeuralNetwork_Calculate()
//    {
//      var net = new SimpleNetwork();
//      net.Build();

//      var res = net.Calculate(new double[] { 1, 2, 3 });

//      Assert.AreEqual(4, res[0]);
//    }

//    [TestMethod]
//    public void NeuralNetwork_TryGetParam()
//    {
//      var net = new SimpleNetwork();
//      net.Build();

//      double par1;
//      double par2;
//      double par3;
//      double par4;
//      double par5;
//      var res1 = net.TryGetParam(0,  out par1);
//      var res2 = net.TryGetParam(3,  out par2);
//      var res3 = net.TryGetParam(9,  out par3);
//      var res4 = net.TryGetParam(12, out par4);
//      var res5 = net.TryGetParam(15, out par5);

//      Assert.IsTrue(res1);
//      Assert.IsTrue(res2);
//      Assert.IsTrue(res3);
//      Assert.IsTrue(res4);
//      Assert.IsFalse(res5);
//      Assert.AreEqual(-1, par1);
//      Assert.AreEqual( 3, par2);
//      Assert.AreEqual( 0, par3);
//      Assert.AreEqual( 1, par4);
//    }

//    [TestMethod]
//    public void NeuralNetwork_TrySetParam()
//    {
//      var net = new SimpleNetwork();
//      net.Build();

//      var res1  = net.TrySetParam(0,   1, false);
//      var res2  = net.TrySetParam(3,   1, true);
//      var res3  = net.TrySetParam(9,  -2, false);
//      var res4  = net.TrySetParam(12, -4, true);
//      var res5  = net.TrySetParam(15,  1, false);

//      Assert.IsTrue(res1);
//      Assert.IsTrue(res2);
//      Assert.IsTrue(res3);
//      Assert.IsTrue(res4);
//      Assert.IsFalse(res5);

//      Assert.AreEqual(1,  net[0][0][0]);
//      Assert.AreEqual(4,  net[0][1][2]);
//      Assert.AreEqual(-2, net[1][0].Bias);
//      Assert.AreEqual(-3, net[2][0][0]);
//    }

//    [TestMethod]
//    public void NeuralNetwork_TryUpdateParams()
//    {
//      var net = new SimpleNetwork();
//      net.Build();

//      var pars = new double[] { 1, 2, 1, 3, -4, 2, -1, 3, 1, -1, 2, 5, 3, 5, 1, -1, 2, -3, 4, 5 };
//      var cursor = 1;

//      var res = net.TryUpdateParams(pars, false, ref cursor);
//      Assert.IsTrue(res);
//      Assert.AreEqual(16, cursor);
//      Assert.AreEqual(2,  net[0][0][0]);
//      Assert.AreEqual(0,  net[0][0][1]);
//      Assert.AreEqual(0,  net[0][0][2]);
//      Assert.AreEqual(1,  net[0][0].Bias);
//      Assert.AreEqual(0,  net[0][1][0]);
//      Assert.AreEqual(3,  net[0][1][1]);
//      Assert.AreEqual(-4, net[0][1][2]);
//      Assert.AreEqual(2,  net[0][1].Bias);
//      Assert.AreEqual(5,  net[1][1][2]);
//      Assert.AreEqual(5,  net[2][0][0]);
//      Assert.AreEqual(1,  net[2][0][1]);

//      cursor = 1;
//      res = net.TryUpdateParams(pars, true, ref cursor);
//      Assert.IsTrue(res);
//      Assert.AreEqual(16, cursor);
//      Assert.AreEqual(4,  net[0][0][0]);
//      Assert.AreEqual(0,  net[0][0][1]);
//      Assert.AreEqual(0,  net[0][0][2]);
//      Assert.AreEqual(2,  net[0][0].Bias);
//      Assert.AreEqual(0,  net[0][1][0]);
//      Assert.AreEqual(6,  net[0][1][1]);
//      Assert.AreEqual(-8, net[0][1][2]);
//      Assert.AreEqual(4,  net[0][1].Bias);
//      Assert.AreEqual(10, net[1][1][2]);
//      Assert.AreEqual(10, net[2][0][0]);
//      Assert.AreEqual(2,  net[2][0][1]);

//      res = net.TryUpdateParams(pars, true, ref cursor);
//      Assert.IsFalse(res);
//    }

//    [TestMethod]
//    public void NeuralNetwork_Bench_Calculate()
//    {
//      var ncount = 1000;
//      var lcount = 10;
//      var net = new LargeNetwork(lcount, ncount);
//      net.Build();

//      var input = new double[ncount];
//      for (int i = 0; i < ncount; i++)
//        input[i] = i;

//      var times = 50;
//      var timer = new Stopwatch();
//      timer.Start();
//      for (int t = 0; t < times; t++)
//        net.Calculate(input);
//      timer.Stop();

//      Console.WriteLine("NeuralNetwork_Bench_Calculate: layers={0} neurons={1} elapsed={2}ms", lcount, ncount, (int)(timer.Elapsed.TotalMilliseconds / times));
//    }

//    [TestMethod]
//    public void NeuralNetwork_Bench_BulkSetVSIndexSet()
//    {
//      var ncount = 1000;
//      var lcount = 10;
//      var pcount = ncount * ncount * lcount;
//      var net = new LargeNetwork(lcount, ncount);
//      net.Build();

//      var pars = new double[pcount];
//      pars[1234] = 1.0D;
//      pars[123404] = -1.0D;
//      pars[1200404] = -2.0D;
//      pars[9700404] = 3.0D;

//      var times = 30;
//      var timer = new Stopwatch();
//      timer.Start();
//      for (int i = 0; i < times; i++)
//      {
//        var cursor = 0;
//        net.TryUpdateParams(pars, true, ref cursor);
//      }
//      timer.Stop();

//      Console.WriteLine("Network Bulk set BM: layers={0} neurons={1} elapsed={2}ms", lcount, ncount, (int)(timer.Elapsed.TotalMilliseconds / times));

//      timer.Reset();
//      timer.Start();
//      for (int i = 0; i < times; i++)
//      {
//        net.TrySetParam(1234, 1.0D, true);
//        net.TrySetParam(123404, -1.0D, true);
//        net.TrySetParam(200404, -2.0D, true);
//        net.TrySetParam(700404, 3.0D, true);
//      }
//      timer.Stop();

//      Console.WriteLine("Network Index set BM: layers={0} neurons={1} elapsed={2}ms", lcount, ncount, Math.Round(timer.Elapsed.TotalMilliseconds / times, 2));
//    }


//    #endregion
//  }
//}
