using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ML.Core;
using ML.Core.ComputingNetworks;
using ML.DeepMethods.Model;

namespace ML.Tests.UnitTests
{
  [TestClass]
  public class ConvolutionNetworkTests : TestBase
  {
    public const double EPS = 0.0000001D;

    #region Inner

    #endregion

    [ClassInitialize]
    public static void ClassInit(TestContext context)
    {
      BaseClassInit(context);
    }

    #region KernelNeuron

    [TestMethod]
    public void KernelNeuron_Build()
    {
      var neuron = new KernelNeuron(1, 15, 10, 7, 4, 1, 1);
      neuron.Build();

      Assert.AreEqual(29, neuron.ParamCount);
      Assert.AreEqual(10, neuron.InputWidth);
      Assert.AreEqual(15, neuron.InputHeight);
      Assert.AreEqual( 4, neuron.WindowWidth);
      Assert.AreEqual( 7, neuron.WindowHeight);
      Assert.AreEqual( 7, neuron.OutputWidth);
      Assert.AreEqual( 9, neuron.OutputHeight);

      Assert.IsNotNull(neuron.NetValues);
      Assert.IsNotNull(neuron.Values);
      Assert.IsNotNull(neuron.Derivatives);

      Assert.AreEqual(9, neuron.NetValues.GetLength(0));
      Assert.AreEqual(7, neuron.NetValues.GetLength(1));
      Assert.AreEqual(9, neuron.Values.GetLength(0));
      Assert.AreEqual(7, neuron.Values.GetLength(1));
      Assert.AreEqual(9, neuron.Derivatives.GetLength(0));
      Assert.AreEqual(7, neuron.Derivatives.GetLength(1));
    }

    [TestMethod]
    public void KernelNeuron_DefaultSteps_Build()
    {
      var neuron = getLargeKernelNeuron();
      neuron.Build();

      Assert.AreEqual(21, neuron.ParamCount);
      Assert.AreEqual(9,  neuron.InputWidth);
      Assert.AreEqual(8,  neuron.InputHeight);
      Assert.AreEqual(5,  neuron.WindowWidth);
      Assert.AreEqual(4,  neuron.WindowHeight);
      Assert.AreEqual(2,  neuron.HeightStride);
      Assert.AreEqual(2,  neuron.WidthStride);
      Assert.AreEqual(3,  neuron.OutputWidth);
      Assert.AreEqual(3,  neuron.OutputHeight);

      Assert.IsNotNull(neuron.NetValues);
      Assert.IsNotNull(neuron.Values);
      Assert.IsNotNull(neuron.Derivatives);

      Assert.AreEqual(3, neuron.NetValues.GetLength(0));
      Assert.AreEqual(3, neuron.NetValues.GetLength(1));
      Assert.AreEqual(3, neuron.Values.GetLength(0));
      Assert.AreEqual(3, neuron.Values.GetLength(1));
      Assert.AreEqual(3, neuron.Derivatives.GetLength(0));
      Assert.AreEqual(3, neuron.Derivatives.GetLength(1));
    }

    [TestMethod]
    public void KernelNeuron_Calculate()
    {
      var neuron = getSimpleKernelNeuron();
      neuron.Build();
      var input = new double[,,]
                  {
                    {
                      { 0, 0, 0, 0, 0 },
                      { 0, 0, 2, 0, 0 },
                      { 0, 1, 2, 0, 0 },
                      { 0, 0, 2, 0, 0 },
                      { 0, 1, 2, 1, 0 },
                      { 0, 2, 2, 2, 0 },
                      { 0, 0, 0, 0, 0 }
                    }
                  };

      var res = neuron.Calculate(input);

      Assert.AreEqual(3, res.GetLength(0));
      Assert.AreEqual(3, res.GetLength(1));
      Assert.AreEqual(0, res[0,0]);
      Assert.AreEqual(6, res[0,1]);
      Assert.AreEqual(5, res[0,2]);
      Assert.AreEqual(7, res[1,0]);
      Assert.AreEqual(7, res[1,1]);
      Assert.AreEqual(5, res[1,2]);
      Assert.AreEqual(3, res[2,0]);
      Assert.AreEqual(5, res[2,1]);
      Assert.AreEqual(4, res[2,2]);

      Assert.AreEqual(0, neuron.NetValues[0,0]);
      Assert.AreEqual(6, neuron.NetValues[0,1]);
      Assert.AreEqual(5, neuron.NetValues[0,2]);
      Assert.AreEqual(7, neuron.NetValues[1,0]);
      Assert.AreEqual(7, neuron.NetValues[1,1]);
      Assert.AreEqual(5, neuron.NetValues[1,2]);
      Assert.AreEqual(3, neuron.NetValues[2,0]);
      Assert.AreEqual(5, neuron.NetValues[2,1]);
      Assert.AreEqual(4, neuron.NetValues[2,2]);

      Assert.AreEqual(0, neuron.Values[0,0]);
      Assert.AreEqual(6, neuron.Values[0,1]);
      Assert.AreEqual(5, neuron.Values[0,2]);
      Assert.AreEqual(7, neuron.Values[1,0]);
      Assert.AreEqual(7, neuron.Values[1,1]);
      Assert.AreEqual(5, neuron.Values[1,2]);
      Assert.AreEqual(3, neuron.Values[2,0]);
      Assert.AreEqual(5, neuron.Values[2,1]);
      Assert.AreEqual(4, neuron.Values[2,2]);

      Assert.AreEqual(1, neuron.Derivatives[0,0]);
      Assert.AreEqual(1, neuron.Derivatives[0,1]);
      Assert.AreEqual(1, neuron.Derivatives[0,2]);
      Assert.AreEqual(1, neuron.Derivatives[1,0]);
      Assert.AreEqual(1, neuron.Derivatives[1,1]);
      Assert.AreEqual(1, neuron.Derivatives[1,2]);
      Assert.AreEqual(1, neuron.Derivatives[2,0]);
      Assert.AreEqual(1, neuron.Derivatives[2,1]);
      Assert.AreEqual(1, neuron.Derivatives[2,2]);
    }

    [TestMethod]
    public void KernelNeuron_DefaultSteps_Calculate()
    {
      var neuron = getLargeKernelNeuron();
      neuron.Build();
      var input = new double[,,]
                  {
                    {
                      { 0, 0, 0, 0, 1, 0, 0, 0, 0 },
                      { 0, 0, 0, 0, 1, 0, 0, 0, 0 },
                      { 0, 0, 0, 0, 1, 0, 0, 0, 0 },
                      { 1, 1, 1, 1, 1, 1, 1, 1, 1 },
                      { 0, 0, 0, 0, 1, 0, 0, 0, 0 },
                      { 0, 0, 0, 0, 1, 0, 0, 0, 0 },
                      { 0, 0, 0, 0, 1, 0, 0, 0, 0 },
                      { 0, 0, 0, 0, 1, 0, 0, 0, 0 }
                    }
                  };

      var res = neuron.Calculate(input);

      Assert.AreEqual(3, res.GetLength(0));
      Assert.AreEqual(3, res.GetLength(1));
      Assert.AreEqual(5, res[0,0]);
      Assert.AreEqual(7, res[0,1]);
      Assert.AreEqual(4, res[0,2]);
      Assert.AreEqual(3, res[1,0]);
      Assert.AreEqual(5, res[1,1]);
      Assert.AreEqual(3, res[1,2]);
      Assert.AreEqual(2, res[2,0]);
      Assert.AreEqual(5, res[2,1]);
      Assert.AreEqual(2, res[2,2]);

      Assert.AreEqual(5, neuron.NetValues[0,0]);
      Assert.AreEqual(7, neuron.NetValues[0,1]);
      Assert.AreEqual(4, neuron.NetValues[0,2]);
      Assert.AreEqual(3, neuron.NetValues[1,0]);
      Assert.AreEqual(5, neuron.NetValues[1,1]);
      Assert.AreEqual(3, neuron.NetValues[1,2]);
      Assert.AreEqual(2, neuron.NetValues[2,0]);
      Assert.AreEqual(5, neuron.NetValues[2,1]);
      Assert.AreEqual(2, neuron.NetValues[2,2]);

      Assert.AreEqual(5, neuron.Values[0,0]);
      Assert.AreEqual(7, neuron.Values[0,1]);
      Assert.AreEqual(4, neuron.Values[0,2]);
      Assert.AreEqual(3, neuron.Values[1,0]);
      Assert.AreEqual(5, neuron.Values[1,1]);
      Assert.AreEqual(3, neuron.Values[1,2]);
      Assert.AreEqual(2, neuron.Values[2,0]);
      Assert.AreEqual(5, neuron.Values[2,1]);
      Assert.AreEqual(2, neuron.Values[2,2]);

      Assert.AreEqual(1, neuron.Derivatives[0,0]);
      Assert.AreEqual(1, neuron.Derivatives[0,1]);
      Assert.AreEqual(1, neuron.Derivatives[0,2]);
      Assert.AreEqual(1, neuron.Derivatives[1,0]);
      Assert.AreEqual(1, neuron.Derivatives[1,1]);
      Assert.AreEqual(1, neuron.Derivatives[1,2]);
      Assert.AreEqual(1, neuron.Derivatives[2,0]);
      Assert.AreEqual(1, neuron.Derivatives[2,1]);
      Assert.AreEqual(1, neuron.Derivatives[2,2]);
    }

    [TestMethod]
    public void KernelNeuron_Calculate_ActivationFunction()
    {
      var neuron = getSimpleKernelNeuron();
      neuron.ActivationFunction = Registry.ActivationFunctions.Exp;
      neuron.Build();
      var input = new double[,,]
                  {
                    {
                      { 0, 0, 0, 0, 0 },
                      { 0, 0, 2, 0, 0 },
                      { 0, 1, 2, 0, 0 },
                      { 0, 0, 2, 0, 0 },
                      { 0, 1, 2, 1, 0 },
                      { 0, 2, 2, 2, 0 },
                      { 0, 0, 0, 0, 0 }
                    }
                  };

      var res = neuron.Calculate(input);

      Assert.AreEqual(3, res.GetLength(0));
      Assert.AreEqual(3, res.GetLength(1));
      Assert.AreEqual(Math.Exp(0), res[0,0], EPS);
      Assert.AreEqual(Math.Exp(6), res[0,1], EPS);
      Assert.AreEqual(Math.Exp(5), res[0,2], EPS);
      Assert.AreEqual(Math.Exp(7), res[1,0], EPS);
      Assert.AreEqual(Math.Exp(7), res[1,1], EPS);
      Assert.AreEqual(Math.Exp(5), res[1,2], EPS);
      Assert.AreEqual(Math.Exp(3), res[2,0], EPS);
      Assert.AreEqual(Math.Exp(5), res[2,1], EPS);
      Assert.AreEqual(Math.Exp(4), res[2,2], EPS);

      Assert.AreEqual(0, neuron.NetValues[0,0]);
      Assert.AreEqual(6, neuron.NetValues[0,1]);
      Assert.AreEqual(5, neuron.NetValues[0,2]);
      Assert.AreEqual(7, neuron.NetValues[1,0]);
      Assert.AreEqual(7, neuron.NetValues[1,1]);
      Assert.AreEqual(5, neuron.NetValues[1,2]);
      Assert.AreEqual(3, neuron.NetValues[2,0]);
      Assert.AreEqual(5, neuron.NetValues[2,1]);
      Assert.AreEqual(4, neuron.NetValues[2,2]);

      Assert.AreEqual(Math.Exp(0), neuron.Values[0,0], EPS);
      Assert.AreEqual(Math.Exp(6), neuron.Values[0,1], EPS);
      Assert.AreEqual(Math.Exp(5), neuron.Values[0,2], EPS);
      Assert.AreEqual(Math.Exp(7), neuron.Values[1,0], EPS);
      Assert.AreEqual(Math.Exp(7), neuron.Values[1,1], EPS);
      Assert.AreEqual(Math.Exp(5), neuron.Values[1,2], EPS);
      Assert.AreEqual(Math.Exp(3), neuron.Values[2,0], EPS);
      Assert.AreEqual(Math.Exp(5), neuron.Values[2,1], EPS);
      Assert.AreEqual(Math.Exp(4), neuron.Values[2,2], EPS);

      Assert.AreEqual(Math.Exp(0), neuron.Derivatives[0,0], EPS);
      Assert.AreEqual(Math.Exp(6), neuron.Derivatives[0,1], EPS);
      Assert.AreEqual(Math.Exp(5), neuron.Derivatives[0,2], EPS);
      Assert.AreEqual(Math.Exp(7), neuron.Derivatives[1,0], EPS);
      Assert.AreEqual(Math.Exp(7), neuron.Derivatives[1,1], EPS);
      Assert.AreEqual(Math.Exp(5), neuron.Derivatives[1,2], EPS);
      Assert.AreEqual(Math.Exp(3), neuron.Derivatives[2,0], EPS);
      Assert.AreEqual(Math.Exp(5), neuron.Derivatives[2,1], EPS);
      Assert.AreEqual(Math.Exp(4), neuron.Derivatives[2,2], EPS);
    }

    [TestMethod]
    public void KernelNeuron_TryGetParam()
    {
      var neuron = getSimpleKernelNeuron();
      neuron.Build();

      double par1;
      double par2;
      double par3;
      double par4;
      double par5;
      double par6;
      double par7;
      double par8;
      var res1 = neuron.TryGetParam(0,  out par1);
      var res2 = neuron.TryGetParam(1,  out par2);
      var res3 = neuron.TryGetParam(2,  out par3);
      var res4 = neuron.TryGetParam(3,  out par4);
      var res5 = neuron.TryGetParam(7,  out par5);
      var res6 = neuron.TryGetParam(12, out par6);
      var res7 = neuron.TryGetParam(15, out par7);
      var res8 = neuron.TryGetParam(16, out par8);

      Assert.IsTrue(res1);
      Assert.AreEqual(0, par1);
      Assert.IsTrue(res2);
      Assert.AreEqual(1, par2);
      Assert.IsTrue(res3);
      Assert.AreEqual(2, par3);
      Assert.IsTrue(res4);
      Assert.AreEqual(1, par4);
      Assert.IsTrue(res5);
      Assert.AreEqual(-1, par5);
      Assert.IsTrue(res6);
      Assert.AreEqual(0, par6);
      Assert.IsTrue(res7);
      Assert.AreEqual(1, par7);
      Assert.IsFalse(res8);
    }

    [TestMethod]
    public void KernelNeuron_TrySetParam()
    {
      var neuron = getSimpleKernelNeuron();
      neuron.Build();

      var res1 = neuron.TrySetParam(0,  1, false);
      var res2 = neuron.TrySetParam(1,  1, true);
      var res3 = neuron.TrySetParam(2, -2, false);
      var res4 = neuron.TrySetParam(3, -4, true);
      var res5 = neuron.TrySetParam(7,  4, false);
      var res6 = neuron.TrySetParam(12, 4, true);
      var res7 = neuron.TrySetParam(15, 4, true);
      var res8 = neuron.TrySetParam(16, 5, false);

      Assert.IsTrue(res1);
      Assert.AreEqual(1, neuron[0]);
      Assert.AreEqual(1, neuron[0, 0]);
      Assert.IsTrue(res2);
      Assert.AreEqual(2, neuron[1]);
      Assert.AreEqual(2, neuron[0, 1]);
      Assert.IsTrue(res3);
      Assert.AreEqual(-2, neuron[2]);
      Assert.AreEqual(-2, neuron[0, 2]);
      Assert.IsTrue(res4);
      Assert.AreEqual(-3, neuron[3]);
      Assert.AreEqual(-3, neuron[1, 0]);
      Assert.IsTrue(res5);
      Assert.AreEqual(4, neuron[7]);
      Assert.AreEqual(4, neuron[2, 1]);
      Assert.IsTrue(res6);
      Assert.AreEqual(4, neuron[12]);
      Assert.AreEqual(4, neuron[4, 0]);
      Assert.IsTrue(res7);
      Assert.AreEqual(5, neuron.Bias);
      Assert.IsFalse(res8);
    }

    [TestMethod]
    public void KernelNeuron_TryUpdateParams()
    {
      var neuron = getSimpleKernelNeuron();
      neuron.Build();

      var pars = new double[] { 1, 2, 3, -4, -1, 3, 1, -1, 2, 5, 3, 5, 1, 1, 2, 3, 7, -5 };
      var cursor = 1;

      var res = neuron.TryUpdateParams(pars, false, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(17, cursor);
      Assert.AreEqual(2,  neuron[0]);
      Assert.AreEqual(2,  neuron[0,0]);
      Assert.AreEqual(3,  neuron[1]);
      Assert.AreEqual(3,  neuron[0,1]);
      Assert.AreEqual(-4, neuron[2]);
      Assert.AreEqual(-4, neuron[0,2]);
      Assert.AreEqual(-1, neuron[3]);
      Assert.AreEqual(-1, neuron[1,0]);
      Assert.AreEqual(3,  neuron[4]);
      Assert.AreEqual(3,  neuron[1,1]);
      Assert.AreEqual(1,  neuron[5]);
      Assert.AreEqual(1,  neuron[1,2]);
      Assert.AreEqual(-1, neuron[6]);
      Assert.AreEqual(-1, neuron[2,0]);
      Assert.AreEqual(2,  neuron[7]);
      Assert.AreEqual(2,  neuron[2,1]);
      Assert.AreEqual(5,  neuron[8]);
      Assert.AreEqual(5,  neuron[2,2]);
      Assert.AreEqual(3,  neuron[9]);
      Assert.AreEqual(3,  neuron[3,0]);
      Assert.AreEqual(5,  neuron[10]);
      Assert.AreEqual(5,  neuron[3,1]);
      Assert.AreEqual(1,  neuron[11]);
      Assert.AreEqual(1,  neuron[3,2]);
      Assert.AreEqual(1,  neuron[12]);
      Assert.AreEqual(1,  neuron[4,0]);
      Assert.AreEqual(2,  neuron[13]);
      Assert.AreEqual(2,  neuron[4,1]);
      Assert.AreEqual(3,  neuron[14]);
      Assert.AreEqual(3,  neuron[4,2]);
      Assert.AreEqual(7,  neuron.Bias);

      cursor = 1;
      res = neuron.TryUpdateParams(pars, true, ref cursor);
      Assert.IsTrue(res);
      Assert.AreEqual(17, cursor);
      Assert.AreEqual(4,  neuron[0]);
      Assert.AreEqual(4,  neuron[0,0]);
      Assert.AreEqual(6,  neuron[1]);
      Assert.AreEqual(6,  neuron[0,1]);
      Assert.AreEqual(-8, neuron[2]);
      Assert.AreEqual(-8, neuron[0,2]);
      Assert.AreEqual(-2, neuron[3]);
      Assert.AreEqual(-2, neuron[1,0]);
      Assert.AreEqual(6,  neuron[4]);
      Assert.AreEqual(6,  neuron[1,1]);
      Assert.AreEqual(2,  neuron[5]);
      Assert.AreEqual(2,  neuron[1,2]);
      Assert.AreEqual(-2, neuron[6]);
      Assert.AreEqual(-2, neuron[2,0]);
      Assert.AreEqual(4,  neuron[7]);
      Assert.AreEqual(4,  neuron[2,1]);
      Assert.AreEqual(10, neuron[8]);
      Assert.AreEqual(10, neuron[2,2]);
      Assert.AreEqual(6,  neuron[9]);
      Assert.AreEqual(6,  neuron[3,0]);
      Assert.AreEqual(10, neuron[10]);
      Assert.AreEqual(10, neuron[3,1]);
      Assert.AreEqual(2,  neuron[11]);
      Assert.AreEqual(2,  neuron[3,2]);
      Assert.AreEqual(2,  neuron[12]);
      Assert.AreEqual(2,  neuron[4,0]);
      Assert.AreEqual(4,  neuron[13]);
      Assert.AreEqual(4,  neuron[4,1]);
      Assert.AreEqual(6,  neuron[14]);
      Assert.AreEqual(6,  neuron[4,2]);
      Assert.AreEqual(14, neuron.Bias);

      res = neuron.TryUpdateParams(pars, true, ref cursor);
      Assert.IsFalse(res);
    }

    #endregion

    #region .pvt

    private KernelNeuron getSimpleKernelNeuron()
    {
      var neuron = new KernelNeuron(1, 7, 5, 5, 3, 1, 1);
      var kernel = new double[]
                   {
                      0,  1,  2,
                      1,  2, -1,
                      2, -1,  0,
                     -1,  0,  0,
                      0,  0,  1,  /* bias */ 1
                   };
      int cursor = 0;
      neuron.TryUpdateParams(kernel, false, ref cursor);

      return neuron;
    }

    private KernelNeuron getLargeKernelNeuron()
    {
      var neuron = new KernelNeuron(1, 8, 9, 4, 5);
      var kernel = new double[]
                   {
                      0, 0, 1, 1, 1,
                      0, 0, 1, 0, 0,
                      0, 0, 1, 0, 0,
                      1, 1, 1, 0, 0,  /* bias */ 1
                   };
      int cursor = 0;
      neuron.TryUpdateParams(kernel, false, ref cursor);

      return neuron;
    }

    #endregion
  }
}
