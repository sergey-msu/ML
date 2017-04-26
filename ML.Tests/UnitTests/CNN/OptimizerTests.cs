using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ML.DeepMethods.Optimizers;

namespace ML.Tests.UnitTests.CNN
{
  [TestClass]
  public class OptimizerTests : TestBase
  {
    #region SGD

    [TestMethod]
    public void SGDOptimizer_SimpleMultivar()
    {
      // arrange
      var func = new Mocks.SimpleMultivar();
      var lr = 0.1D;
      var w = new double[2][] { new[] { 1.0D, 1.0D }, new[] { 1.0D } };
      var optimizer = new SGDOptimizer();

      // act & assert

      optimizer.Push(w, func.Gradient(w), lr);
      Assert.AreEqual(0,   w[0][0]);
      Assert.AreEqual(0.8, w[0][1]);
      Assert.AreEqual(0,   w[1][0]);
      Assert.AreEqual(2.04, optimizer.Step2);

      optimizer.Push(w, func.Gradient(w), lr);
      Assert.AreEqual(0.16, w[0][0], EPS);
      Assert.AreEqual(0.32, w[0][1], EPS);
      Assert.AreEqual(0.16, w[1][0], EPS);
      Assert.AreEqual(0.2816, optimizer.Step2, EPS);

      optimizer.Push(w, func.Gradient(w), lr);
      Assert.AreEqual(0.032, w[0][0], EPS);
      Assert.AreEqual(0.192, w[0][1], EPS);
      Assert.AreEqual(0.032, w[1][0], EPS);
      Assert.AreEqual(0.049152, optimizer.Step2, EPS);

      optimizer.Push(w, func.Gradient(w), lr);
      Assert.AreEqual(0.032,  w[0][0], EPS);
      Assert.AreEqual(0.0896, w[0][1], EPS);
      Assert.AreEqual(0.032,  w[1][0], EPS);
      Assert.AreEqual(0.01048576, optimizer.Step2, EPS);

      optimizer.Push(w, func.Gradient(w), lr);
      Assert.AreEqual(0.01152, w[0][0], EPS);
      Assert.AreEqual(0.04864, w[0][1], EPS);
      Assert.AreEqual(0.01152, w[1][0], EPS);
      Assert.AreEqual(0.00251658, optimizer.Step2, EPS);
    }

    #endregion

    #region Momentum

    [TestMethod]
    public void MomentumOptimizer_SimpleMultivar()
    {
      // arrange
      var func = new Mocks.SimpleMultivar();
      var lr = 0.1D;
      var mu = 0.9D;
      var w  = new double[2][] { new[] { 1.0D, 1.0D }, new[] { 1.0D } };
      var optimizer = new MomentumOptimizer(mu);

      // act & assert

      optimizer.Push(w, func.Gradient(w), lr);
      Assert.AreEqual(0,   w[0][0]);
      Assert.AreEqual(0.8, w[0][1]);
      Assert.AreEqual(0,   w[1][0]);
      Assert.AreEqual(2.04, optimizer.Step2, EPS);

      optimizer.Push(w, func.Gradient(w), lr);
      Assert.AreEqual(-0.74, w[0][0], EPS);
      Assert.AreEqual( 0.14, w[0][1], EPS);
      Assert.AreEqual(-0.74, w[1][0], EPS);
      Assert.AreEqual(1.5308, optimizer.Step2, EPS);

      optimizer.Push(w, func.Gradient(w), lr);
      Assert.AreEqual(-0.490, w[0][0], EPS);
      Assert.AreEqual(-0.834, w[0][1], EPS);
      Assert.AreEqual(-0.490, w[1][0], EPS);
      Assert.AreEqual(1.073676, optimizer.Step2, EPS);

      optimizer.Push(w, func.Gradient(w), lr);
      Assert.AreEqual( 0.1562, w[0][0], EPS);
      Assert.AreEqual(-1.4062, w[0][1], EPS);
      Assert.AreEqual( 0.1562, w[1][0], EPS);
      Assert.AreEqual(1.16256172, optimizer.Step2, EPS);

      optimizer.Push(w, func.Gradient(w), lr);
      Assert.AreEqual( 0.2691,  w[0][0], EPS);
      Assert.AreEqual(-1.01498, w[0][1], EPS);
      Assert.AreEqual( 0.2691,  w[1][0], EPS);
      Assert.AreEqual(0.17854591, optimizer.Step2, EPS);
    }

    #endregion

    #region Adagrad

    [TestMethod]
    public void AdagradOptimizer_SimpleMultivar()
    {
      // arrange
      var func = new Mocks.SimpleMultivar();
      var lr  = 0.1D;
      var eps = 0.2D;
      var w   = new double[2][] { new[] { 1.0D, 1.0D }, new[] { 1.0D } };
      var optimizer = new AdagradOptimizer(eps);

      // act & assert

      optimizer.Push(w, func.Gradient(w), lr);
      Assert.AreEqual(0.90009985, w[0][0], EPS);
      Assert.AreEqual(0.90240999, w[0][1], EPS);
      Assert.AreEqual(0.90009985, w[1][0], EPS);
      Assert.AreEqual(0.02948389, optimizer.Step2, EPS);

      optimizer.Push(w, func.Gradient(w), lr);
      Assert.AreEqual(0.83325519, w[0][0], EPS);
      Assert.AreEqual(0.83612927, w[0][1], EPS);
      Assert.AreEqual(0.83325519, w[1][0], EPS);
      Assert.AreEqual(0.01332955, optimizer.Step2, EPS);

      optimizer.Push(w, func.Gradient(w), lr);
      Assert.AreEqual(0.78064134, w[0][0], EPS);
      Assert.AreEqual(0.78373111, w[0][1], EPS);
      Assert.AreEqual(0.78064134, w[1][0], EPS);
      Assert.AreEqual(0.008282, optimizer.Step2, EPS);

      optimizer.Push(w, func.Gradient(w), lr);
      Assert.AreEqual(0.73643259, w[0][0], EPS);
      Assert.AreEqual(0.73961111, w[0][1], EPS);
      Assert.AreEqual(0.73643259, w[1][0], EPS);
      Assert.AreEqual(0.0058554, optimizer.Step2, EPS);

      optimizer.Push(w, func.Gradient(w), lr);
      Assert.AreEqual(0.69794315, w[0][0], EPS);
      Assert.AreEqual(0.70115040, w[0][1], EPS);
      Assert.AreEqual(0.69794315, w[1][0], EPS);
      Assert.AreEqual(0.0044421, optimizer.Step2, EPS);
    }

    #endregion

    #region RMSprop

    [TestMethod]
    public void RMSPropOptimizer_SimpleMultivar()
    {
      // arrange
      var func  = new Mocks.SimpleMultivar();
      var lr    = 0.1D;
      var eps   = 0.2D;
      var gamma = 0.4D;
      var w     = new double[2][] { new[] { 1.0D, 1.0D }, new[] { 1.0D } };
      var optimizer = new RMSPropOptimizer(eps, gamma);

      // act & assert

      optimizer.Push(w, func.Gradient(w), lr);
      Assert.AreEqual(0.87111518, w[0][0], EPS);
      Assert.AreEqual(0.87596527, w[0][1], EPS);
      Assert.AreEqual(0.87111518, w[1][0], EPS);
      Assert.AreEqual(0.04860721, optimizer.Step2, EPS);

      optimizer.Push(w, func.Gradient(w), lr);
      Assert.AreEqual(0.76683636, w[0][0], EPS);
      Assert.AreEqual(0.77441535, w[0][1], EPS);
      Assert.AreEqual(0.76683636, w[1][0], EPS);
      Assert.AreEqual(0.03206053, optimizer.Step2, EPS);

      optimizer.Push(w, func.Gradient(w), lr);
      Assert.AreEqual(0.67050260, w[0][0], EPS);
      Assert.AreEqual(0.68059867, w[0][1], EPS);
      Assert.AreEqual(0.67050260, w[1][0], EPS);
      Assert.AreEqual(0.02736195, optimizer.Step2, EPS);

      optimizer.Push(w, func.Gradient(w), lr);
      Assert.AreEqual(0.57795893, w[0][0], EPS);
      Assert.AreEqual(0.59072338, w[0][1], EPS);
      Assert.AreEqual(0.57795893, w[1][0], EPS);
      Assert.AreEqual(0.02520623, optimizer.Step2, EPS);

      optimizer.Push(w, func.Gradient(w), lr);
      Assert.AreEqual(0.48793820, w[0][0], EPS);
      Assert.AreEqual(0.50366403, w[0][1], EPS);
      Assert.AreEqual(0.48793820, w[1][0], EPS);
      Assert.AreEqual(0.02378680, optimizer.Step2, EPS);
    }

    #endregion

    #region Adadelta

    [TestMethod]
    public void AdadeltaOptimizer_SimpleMultivar()
    {
      // arrange
      var func  = new Mocks.SimpleMultivar();
      var lr    = 1.0D;
      var eps   = 0.01D;
      var gamma = 0.9D;
      var w     = new double[2][] { new[] { 1.0D, 1.0D }, new[] { 1.0D } };
      var optimizer = new AdadeltaOptimizer(eps, gamma, true);

      // act & assert

      optimizer.Push(w, func.Gradient(w), lr);
      Assert.AreEqual(0.68393023, w[0][0], EPS);
      Assert.AreEqual(0.68765248, w[0][1], EPS);
      Assert.AreEqual(0.68393023, w[1][0], EPS);
      Assert.AreEqual(0.29736118, optimizer.Step2, EPS);

      optimizer.Push(w, func.Gradient(w), lr);
      Assert.AreEqual(0.42274831, w[0][0], EPS);
      Assert.AreEqual(0.42729503, w[0][1], EPS);
      Assert.AreEqual(0.42274831, w[1][0], EPS);
      Assert.AreEqual(0.20421799, optimizer.Step2, EPS);

      optimizer.Push(w, func.Gradient(w), lr);
      Assert.AreEqual(0.24219537, w[0][0], EPS);
      Assert.AreEqual(0.24472560, w[0][1], EPS);
      Assert.AreEqual(0.24219537, w[1][0], EPS);
      Assert.AreEqual(0.09853033, optimizer.Step2, EPS);

      optimizer.Push(w, func.Gradient(w), lr);
      Assert.AreEqual(0.13217318, w[0][0], EPS);
      Assert.AreEqual(0.13351822, w[0][1], EPS);
      Assert.AreEqual(0.13217318, w[1][0], EPS);
      Assert.AreEqual(0.03657684, optimizer.Step2, EPS);

      optimizer.Push(w, func.Gradient(w), lr);
      Assert.AreEqual(0.06995692, w[0][0], EPS);
      Assert.AreEqual(0.07071067, w[0][1], EPS);
      Assert.AreEqual(0.06995692, w[1][0], EPS);
      Assert.AreEqual(0.01168651, optimizer.Step2, EPS);
    }

    #endregion

    #region Adam

    [TestMethod]
    public void AdamOptimizer_SimpleMultivar()
    {
      // arrange
      var func  = new Mocks.SimpleMultivar();
      var lr    = 0.1D;
      var eps   = 0.2D;
      var beta1 = 0.9D;
      var beta2 = 0.7D;
      var w     = new double[2][] { new[] { 1.0D, 1.0D }, new[] { 1.0D } };
      var optimizer = new AdamOptimizer(beta1, beta2, eps);

      // act & assert

      optimizer.Push(w, func.Gradient(w), lr);
      Assert.AreEqual(0.90196078, w[0][0], EPS);
      Assert.AreEqual(0.90909091, w[0][1], EPS);
      Assert.AreEqual(0.90196078, w[1][0], EPS);
      Assert.AreEqual(0.02748784, optimizer.Step2, EPS);

      optimizer.Push(w, func.Gradient(w), lr);
      Assert.AreEqual(0.80353053, w[0][0], EPS);
      Assert.AreEqual(0.81818465, w[0][1], EPS);
      Assert.AreEqual(0.80353053, w[1][0], EPS);
      Assert.AreEqual(0.02764098, optimizer.Step2, EPS);

      optimizer.Push(w, func.Gradient(w), lr);
      Assert.AreEqual(0.70434233, w[0][0], EPS);
      Assert.AreEqual(0.72707454, w[0][1], EPS);
      Assert.AreEqual(0.70434233, w[1][0], EPS);
      Assert.AreEqual(0.02797765, optimizer.Step2, EPS);

      optimizer.Push(w, func.Gradient(w), lr);
      Assert.AreEqual(0.60398299, w[0][0], EPS);
      Assert.AreEqual(0.63554944, w[0][1], EPS);
      Assert.AreEqual(0.60398299, w[1][0], EPS);
      Assert.AreEqual(0.02852083, optimizer.Step2, EPS);

      optimizer.Push(w, func.Gradient(w), lr);
      Assert.AreEqual(0.50198181, w[0][0], EPS);
      Assert.AreEqual(0.54339582, w[0][1], EPS);
      Assert.AreEqual(0.50198181, w[1][0], EPS);
      Assert.AreEqual(0.02930077, optimizer.Step2, EPS);
    }

    #endregion

    //   Adamax
  }
}
