//using System;
//using Microsoft.VisualStudio.TestTools.UnitTesting;
//using ML.Core;
//using ML.Core.ComputingNetworks;
//using ML.DeepMethods.Model;

//namespace ML.Tests.UnitTests
//{
//  [TestClass]
//  public class ConvolutionNetworkTests : TestBase
//  {
//    public const double EPS = 0.0000001D;

//    #region Inner

//    #endregion

//    [ClassInitialize]
//    public static void ClassInit(TestContext context)
//    {
//      BaseClassInit(context);
//    }

//    #region FeatureMap

//    [TestMethod]
//    public void Kernelfm_Build()
//    {
//      var fm = new FeatureMap(1, 15, 10, 7, 4, 1, 1);
//      fm.Build();

//      Assert.AreEqual(29, fm.ParamCount);
//      Assert.AreEqual(10, fm.InputWidth);
//      Assert.AreEqual(15, fm.InputSize);
//      Assert.AreEqual( 4, fm.WindowWidth);
//      Assert.AreEqual( 7, fm.WindowSize);
//      Assert.AreEqual( 7, fm.OutputWidth);
//      Assert.AreEqual( 9, fm.OutputSize);

//      Assert.IsNotNull(fm.NetValues);
//      Assert.IsNotNull(fm.Values);
//      Assert.IsNotNull(fm.Derivatives);

//      Assert.AreEqual(9, fm.NetValues.GetLength(0));
//      Assert.AreEqual(7, fm.NetValues.GetLength(1));
//      Assert.AreEqual(9, fm.Values.GetLength(0));
//      Assert.AreEqual(7, fm.Values.GetLength(1));
//      Assert.AreEqual(9, fm.Derivatives.GetLength(0));
//      Assert.AreEqual(7, fm.Derivatives.GetLength(1));
//    }

//    [TestMethod]
//    public void Kernelfm_DefaultSteps_Build()
//    {
//      var fm = getLargeKernelfm();
//      fm.Build();

//      Assert.AreEqual(21, fm.ParamCount);
//      Assert.AreEqual(9,  fm.InputWidth);
//      Assert.AreEqual(8,  fm.InputSize);
//      Assert.AreEqual(5,  fm.WindowWidth);
//      Assert.AreEqual(4,  fm.WindowSize);
//      Assert.AreEqual(2,  fm.Stride);
//      Assert.AreEqual(2,  fm.WidthStride);
//      Assert.AreEqual(3,  fm.OutputWidth);
//      Assert.AreEqual(3,  fm.OutputSize);

//      Assert.IsNotNull(fm.NetValues);
//      Assert.IsNotNull(fm.Values);
//      Assert.IsNotNull(fm.Derivatives);

//      Assert.AreEqual(3, fm.NetValues.GetLength(0));
//      Assert.AreEqual(3, fm.NetValues.GetLength(1));
//      Assert.AreEqual(3, fm.Values.GetLength(0));
//      Assert.AreEqual(3, fm.Values.GetLength(1));
//      Assert.AreEqual(3, fm.Derivatives.GetLength(0));
//      Assert.AreEqual(3, fm.Derivatives.GetLength(1));
//    }

//    [TestMethod]
//    public void Kernelfm_Calculate()
//    {
//      var fm = getSimpleKernelfm();
//      fm.Build();
//      var input = new double[,,]
//                  {
//                    {
//                      { 0, 0, 0, 0, 0 },
//                      { 0, 0, 2, 0, 0 },
//                      { 0, 1, 2, 0, 0 },
//                      { 0, 0, 2, 0, 0 },
//                      { 0, 1, 2, 1, 0 },
//                      { 0, 2, 2, 2, 0 },
//                      { 0, 0, 0, 0, 0 }
//                    }
//                  };

//      var res = fm.Calculate(input);

//      Assert.AreEqual(3, res.GetLength(0));
//      Assert.AreEqual(3, res.GetLength(1));
//      Assert.AreEqual(0, res[0,0]);
//      Assert.AreEqual(6, res[0,1]);
//      Assert.AreEqual(5, res[0,2]);
//      Assert.AreEqual(7, res[1,0]);
//      Assert.AreEqual(7, res[1,1]);
//      Assert.AreEqual(5, res[1,2]);
//      Assert.AreEqual(3, res[2,0]);
//      Assert.AreEqual(5, res[2,1]);
//      Assert.AreEqual(4, res[2,2]);

//      Assert.AreEqual(0, fm.NetValues[0,0]);
//      Assert.AreEqual(6, fm.NetValues[0,1]);
//      Assert.AreEqual(5, fm.NetValues[0,2]);
//      Assert.AreEqual(7, fm.NetValues[1,0]);
//      Assert.AreEqual(7, fm.NetValues[1,1]);
//      Assert.AreEqual(5, fm.NetValues[1,2]);
//      Assert.AreEqual(3, fm.NetValues[2,0]);
//      Assert.AreEqual(5, fm.NetValues[2,1]);
//      Assert.AreEqual(4, fm.NetValues[2,2]);

//      Assert.AreEqual(0, fm.Values[0,0]);
//      Assert.AreEqual(6, fm.Values[0,1]);
//      Assert.AreEqual(5, fm.Values[0,2]);
//      Assert.AreEqual(7, fm.Values[1,0]);
//      Assert.AreEqual(7, fm.Values[1,1]);
//      Assert.AreEqual(5, fm.Values[1,2]);
//      Assert.AreEqual(3, fm.Values[2,0]);
//      Assert.AreEqual(5, fm.Values[2,1]);
//      Assert.AreEqual(4, fm.Values[2,2]);

//      Assert.AreEqual(1, fm.Derivatives[0,0]);
//      Assert.AreEqual(1, fm.Derivatives[0,1]);
//      Assert.AreEqual(1, fm.Derivatives[0,2]);
//      Assert.AreEqual(1, fm.Derivatives[1,0]);
//      Assert.AreEqual(1, fm.Derivatives[1,1]);
//      Assert.AreEqual(1, fm.Derivatives[1,2]);
//      Assert.AreEqual(1, fm.Derivatives[2,0]);
//      Assert.AreEqual(1, fm.Derivatives[2,1]);
//      Assert.AreEqual(1, fm.Derivatives[2,2]);
//    }

//    [TestMethod]
//    public void Kernelfm_DefaultSteps_Calculate()
//    {
//      var fm = getLargeKernelfm();
//      fm.Build();
//      var input = new double[,,]
//                  {
//                    {
//                      { 0, 0, 0, 0, 1, 0, 0, 0, 0 },
//                      { 0, 0, 0, 0, 1, 0, 0, 0, 0 },
//                      { 0, 0, 0, 0, 1, 0, 0, 0, 0 },
//                      { 1, 1, 1, 1, 1, 1, 1, 1, 1 },
//                      { 0, 0, 0, 0, 1, 0, 0, 0, 0 },
//                      { 0, 0, 0, 0, 1, 0, 0, 0, 0 },
//                      { 0, 0, 0, 0, 1, 0, 0, 0, 0 },
//                      { 0, 0, 0, 0, 1, 0, 0, 0, 0 }
//                    }
//                  };

//      var res = fm.Calculate(input);

//      Assert.AreEqual(3, res.GetLength(0));
//      Assert.AreEqual(3, res.GetLength(1));
//      Assert.AreEqual(5, res[0,0]);
//      Assert.AreEqual(7, res[0,1]);
//      Assert.AreEqual(4, res[0,2]);
//      Assert.AreEqual(3, res[1,0]);
//      Assert.AreEqual(5, res[1,1]);
//      Assert.AreEqual(3, res[1,2]);
//      Assert.AreEqual(2, res[2,0]);
//      Assert.AreEqual(5, res[2,1]);
//      Assert.AreEqual(2, res[2,2]);

//      Assert.AreEqual(5, fm.NetValues[0,0]);
//      Assert.AreEqual(7, fm.NetValues[0,1]);
//      Assert.AreEqual(4, fm.NetValues[0,2]);
//      Assert.AreEqual(3, fm.NetValues[1,0]);
//      Assert.AreEqual(5, fm.NetValues[1,1]);
//      Assert.AreEqual(3, fm.NetValues[1,2]);
//      Assert.AreEqual(2, fm.NetValues[2,0]);
//      Assert.AreEqual(5, fm.NetValues[2,1]);
//      Assert.AreEqual(2, fm.NetValues[2,2]);

//      Assert.AreEqual(5, fm.Values[0,0]);
//      Assert.AreEqual(7, fm.Values[0,1]);
//      Assert.AreEqual(4, fm.Values[0,2]);
//      Assert.AreEqual(3, fm.Values[1,0]);
//      Assert.AreEqual(5, fm.Values[1,1]);
//      Assert.AreEqual(3, fm.Values[1,2]);
//      Assert.AreEqual(2, fm.Values[2,0]);
//      Assert.AreEqual(5, fm.Values[2,1]);
//      Assert.AreEqual(2, fm.Values[2,2]);

//      Assert.AreEqual(1, fm.Derivatives[0,0]);
//      Assert.AreEqual(1, fm.Derivatives[0,1]);
//      Assert.AreEqual(1, fm.Derivatives[0,2]);
//      Assert.AreEqual(1, fm.Derivatives[1,0]);
//      Assert.AreEqual(1, fm.Derivatives[1,1]);
//      Assert.AreEqual(1, fm.Derivatives[1,2]);
//      Assert.AreEqual(1, fm.Derivatives[2,0]);
//      Assert.AreEqual(1, fm.Derivatives[2,1]);
//      Assert.AreEqual(1, fm.Derivatives[2,2]);
//    }

//    [TestMethod]
//    public void Kernelfm_Calculate_ActivationFunction()
//    {
//      var fm = getSimpleKernelfm();
//      fm.ActivationFunction = Registry.ActivationFunctions.Exp;
//      fm.Build();
//      var input = new double[,,]
//                  {
//                    {
//                      { 0, 0, 0, 0, 0 },
//                      { 0, 0, 2, 0, 0 },
//                      { 0, 1, 2, 0, 0 },
//                      { 0, 0, 2, 0, 0 },
//                      { 0, 1, 2, 1, 0 },
//                      { 0, 2, 2, 2, 0 },
//                      { 0, 0, 0, 0, 0 }
//                    }
//                  };

//      var res = fm.Calculate(input);

//      Assert.AreEqual(3, res.GetLength(0));
//      Assert.AreEqual(3, res.GetLength(1));
//      Assert.AreEqual(Math.Exp(0), res[0,0], EPS);
//      Assert.AreEqual(Math.Exp(6), res[0,1], EPS);
//      Assert.AreEqual(Math.Exp(5), res[0,2], EPS);
//      Assert.AreEqual(Math.Exp(7), res[1,0], EPS);
//      Assert.AreEqual(Math.Exp(7), res[1,1], EPS);
//      Assert.AreEqual(Math.Exp(5), res[1,2], EPS);
//      Assert.AreEqual(Math.Exp(3), res[2,0], EPS);
//      Assert.AreEqual(Math.Exp(5), res[2,1], EPS);
//      Assert.AreEqual(Math.Exp(4), res[2,2], EPS);

//      Assert.AreEqual(0, fm.NetValues[0,0]);
//      Assert.AreEqual(6, fm.NetValues[0,1]);
//      Assert.AreEqual(5, fm.NetValues[0,2]);
//      Assert.AreEqual(7, fm.NetValues[1,0]);
//      Assert.AreEqual(7, fm.NetValues[1,1]);
//      Assert.AreEqual(5, fm.NetValues[1,2]);
//      Assert.AreEqual(3, fm.NetValues[2,0]);
//      Assert.AreEqual(5, fm.NetValues[2,1]);
//      Assert.AreEqual(4, fm.NetValues[2,2]);

//      Assert.AreEqual(Math.Exp(0), fm.Values[0,0], EPS);
//      Assert.AreEqual(Math.Exp(6), fm.Values[0,1], EPS);
//      Assert.AreEqual(Math.Exp(5), fm.Values[0,2], EPS);
//      Assert.AreEqual(Math.Exp(7), fm.Values[1,0], EPS);
//      Assert.AreEqual(Math.Exp(7), fm.Values[1,1], EPS);
//      Assert.AreEqual(Math.Exp(5), fm.Values[1,2], EPS);
//      Assert.AreEqual(Math.Exp(3), fm.Values[2,0], EPS);
//      Assert.AreEqual(Math.Exp(5), fm.Values[2,1], EPS);
//      Assert.AreEqual(Math.Exp(4), fm.Values[2,2], EPS);

//      Assert.AreEqual(Math.Exp(0), fm.Derivatives[0,0], EPS);
//      Assert.AreEqual(Math.Exp(6), fm.Derivatives[0,1], EPS);
//      Assert.AreEqual(Math.Exp(5), fm.Derivatives[0,2], EPS);
//      Assert.AreEqual(Math.Exp(7), fm.Derivatives[1,0], EPS);
//      Assert.AreEqual(Math.Exp(7), fm.Derivatives[1,1], EPS);
//      Assert.AreEqual(Math.Exp(5), fm.Derivatives[1,2], EPS);
//      Assert.AreEqual(Math.Exp(3), fm.Derivatives[2,0], EPS);
//      Assert.AreEqual(Math.Exp(5), fm.Derivatives[2,1], EPS);
//      Assert.AreEqual(Math.Exp(4), fm.Derivatives[2,2], EPS);
//    }

//    [TestMethod]
//    public void Kernelfm_TryGetParam()
//    {
//      var fm = getSimpleKernelfm();
//      fm.Build();

//      double par1;
//      double par2;
//      double par3;
//      double par4;
//      double par5;
//      double par6;
//      double par7;
//      double par8;
//      var res1 = fm.TryGetParam(0,  out par1);
//      var res2 = fm.TryGetParam(1,  out par2);
//      var res3 = fm.TryGetParam(2,  out par3);
//      var res4 = fm.TryGetParam(3,  out par4);
//      var res5 = fm.TryGetParam(7,  out par5);
//      var res6 = fm.TryGetParam(12, out par6);
//      var res7 = fm.TryGetParam(15, out par7);
//      var res8 = fm.TryGetParam(16, out par8);

//      Assert.IsTrue(res1);
//      Assert.AreEqual(0, par1);
//      Assert.IsTrue(res2);
//      Assert.AreEqual(1, par2);
//      Assert.IsTrue(res3);
//      Assert.AreEqual(2, par3);
//      Assert.IsTrue(res4);
//      Assert.AreEqual(1, par4);
//      Assert.IsTrue(res5);
//      Assert.AreEqual(-1, par5);
//      Assert.IsTrue(res6);
//      Assert.AreEqual(0, par6);
//      Assert.IsTrue(res7);
//      Assert.AreEqual(1, par7);
//      Assert.IsFalse(res8);
//    }

//    [TestMethod]
//    public void Kernelfm_TrySetParam()
//    {
//      var fm = getSimpleKernelfm();
//      fm.Build();

//      var res1 = fm.TrySetParam(0,  1, false);
//      var res2 = fm.TrySetParam(1,  1, true);
//      var res3 = fm.TrySetParam(2, -2, false);
//      var res4 = fm.TrySetParam(3, -4, true);
//      var res5 = fm.TrySetParam(7,  4, false);
//      var res6 = fm.TrySetParam(12, 4, true);
//      var res7 = fm.TrySetParam(15, 4, true);
//      var res8 = fm.TrySetParam(16, 5, false);

//      Assert.IsTrue(res1);
//      Assert.AreEqual(1, fm[0]);
//      Assert.AreEqual(1, fm[0, 0]);
//      Assert.IsTrue(res2);
//      Assert.AreEqual(2, fm[1]);
//      Assert.AreEqual(2, fm[0, 1]);
//      Assert.IsTrue(res3);
//      Assert.AreEqual(-2, fm[2]);
//      Assert.AreEqual(-2, fm[0, 2]);
//      Assert.IsTrue(res4);
//      Assert.AreEqual(-3, fm[3]);
//      Assert.AreEqual(-3, fm[1, 0]);
//      Assert.IsTrue(res5);
//      Assert.AreEqual(4, fm[7]);
//      Assert.AreEqual(4, fm[2, 1]);
//      Assert.IsTrue(res6);
//      Assert.AreEqual(4, fm[12]);
//      Assert.AreEqual(4, fm[4, 0]);
//      Assert.IsTrue(res7);
//      Assert.AreEqual(5, fm.Bias);
//      Assert.IsFalse(res8);
//    }

//    [TestMethod]
//    public void Kernelfm_TryUpdateParams()
//    {
//      var fm = getSimpleKernelfm();
//      fm.Build();

//      var pars = new double[] { 1, 2, 3, -4, -1, 3, 1, -1, 2, 5, 3, 5, 1, 1, 2, 3, 7, -5 };
//      var cursor = 1;

//      var res = fm.TryUpdateParams(pars, false, ref cursor);
//      Assert.IsTrue(res);
//      Assert.AreEqual(17, cursor);
//      Assert.AreEqual(2,  fm[0]);
//      Assert.AreEqual(2,  fm[0,0]);
//      Assert.AreEqual(3,  fm[1]);
//      Assert.AreEqual(3,  fm[0,1]);
//      Assert.AreEqual(-4, fm[2]);
//      Assert.AreEqual(-4, fm[0,2]);
//      Assert.AreEqual(-1, fm[3]);
//      Assert.AreEqual(-1, fm[1,0]);
//      Assert.AreEqual(3,  fm[4]);
//      Assert.AreEqual(3,  fm[1,1]);
//      Assert.AreEqual(1,  fm[5]);
//      Assert.AreEqual(1,  fm[1,2]);
//      Assert.AreEqual(-1, fm[6]);
//      Assert.AreEqual(-1, fm[2,0]);
//      Assert.AreEqual(2,  fm[7]);
//      Assert.AreEqual(2,  fm[2,1]);
//      Assert.AreEqual(5,  fm[8]);
//      Assert.AreEqual(5,  fm[2,2]);
//      Assert.AreEqual(3,  fm[9]);
//      Assert.AreEqual(3,  fm[3,0]);
//      Assert.AreEqual(5,  fm[10]);
//      Assert.AreEqual(5,  fm[3,1]);
//      Assert.AreEqual(1,  fm[11]);
//      Assert.AreEqual(1,  fm[3,2]);
//      Assert.AreEqual(1,  fm[12]);
//      Assert.AreEqual(1,  fm[4,0]);
//      Assert.AreEqual(2,  fm[13]);
//      Assert.AreEqual(2,  fm[4,1]);
//      Assert.AreEqual(3,  fm[14]);
//      Assert.AreEqual(3,  fm[4,2]);
//      Assert.AreEqual(7,  fm.Bias);

//      cursor = 1;
//      res = fm.TryUpdateParams(pars, true, ref cursor);
//      Assert.IsTrue(res);
//      Assert.AreEqual(17, cursor);
//      Assert.AreEqual(4,  fm[0]);
//      Assert.AreEqual(4,  fm[0,0]);
//      Assert.AreEqual(6,  fm[1]);
//      Assert.AreEqual(6,  fm[0,1]);
//      Assert.AreEqual(-8, fm[2]);
//      Assert.AreEqual(-8, fm[0,2]);
//      Assert.AreEqual(-2, fm[3]);
//      Assert.AreEqual(-2, fm[1,0]);
//      Assert.AreEqual(6,  fm[4]);
//      Assert.AreEqual(6,  fm[1,1]);
//      Assert.AreEqual(2,  fm[5]);
//      Assert.AreEqual(2,  fm[1,2]);
//      Assert.AreEqual(-2, fm[6]);
//      Assert.AreEqual(-2, fm[2,0]);
//      Assert.AreEqual(4,  fm[7]);
//      Assert.AreEqual(4,  fm[2,1]);
//      Assert.AreEqual(10, fm[8]);
//      Assert.AreEqual(10, fm[2,2]);
//      Assert.AreEqual(6,  fm[9]);
//      Assert.AreEqual(6,  fm[3,0]);
//      Assert.AreEqual(10, fm[10]);
//      Assert.AreEqual(10, fm[3,1]);
//      Assert.AreEqual(2,  fm[11]);
//      Assert.AreEqual(2,  fm[3,2]);
//      Assert.AreEqual(2,  fm[12]);
//      Assert.AreEqual(2,  fm[4,0]);
//      Assert.AreEqual(4,  fm[13]);
//      Assert.AreEqual(4,  fm[4,1]);
//      Assert.AreEqual(6,  fm[14]);
//      Assert.AreEqual(6,  fm[4,2]);
//      Assert.AreEqual(14, fm.Bias);

//      res = fm.TryUpdateParams(pars, true, ref cursor);
//      Assert.IsFalse(res);
//    }

//    #endregion

//    #region .pvt

//    private FeatureMap getSimpleKernelfm()
//    {
//      var fm = new FeatureMap(1, 7, 5, 5, 3, 1, 1);
//      var kernel = new double[]
//                   {
//                      0,  1,  2,
//                      1,  2, -1,
//                      2, -1,  0,
//                     -1,  0,  0,
//                      0,  0,  1,  /* bias */ 1
//                   };
//      int cursor = 0;
//      fm.TryUpdateParams(kernel, false, ref cursor);

//      return fm;
//    }

//    private FeatureMap getLargeKernelfm()
//    {
//      var fm = new FeatureMap(1, 8, 9, 4, 5);
//      var kernel = new double[]
//                   {
//                      0, 0, 1, 1, 1,
//                      0, 0, 1, 0, 0,
//                      0, 0, 1, 0, 0,
//                      1, 1, 1, 0, 0,  /* bias */ 1
//                   };
//      int cursor = 0;
//      fm.TryUpdateParams(kernel, false, ref cursor);

//      return fm;
//    }

//    #endregion
//  }
//}
