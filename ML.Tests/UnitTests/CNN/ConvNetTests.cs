using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ML.Core;
using ML.DeepMethods.Models;
using ML.Core.Registry;
using ML.Contracts;

namespace ML.Tests.UnitTests.CNN
{
  [TestClass]
  public class ConvNetTests : TestBase
  {
    #region Inner

    public class LayerMock : DeepLayerBase
    {
      public LayerMock(IActivationFunction activation)
        : base (outputDepth: 1,
                windowSize: 1,
                stride: 1,
                padding: 0,
                activation: activation)
      {
      }


      protected override void DoBackprop(DeepLayerBase prevLayer, double[][,] prevValues, double[][,] prevError, double[][,] errors)
      {
        throw new NotImplementedException();
      }

      protected override void DoCalculate(double[][,] input, double[][,] result)
      {
        throw new NotImplementedException();
      }

      protected override void DoSetLayerGradient(double[][,] prevValues, double[][,] errors, double[] gradient, bool isDelta)
      {
        throw new NotImplementedException();
      }
    }


    #endregion


    [ClassInitialize]
    public static void ClassInit(TestContext context)
    {
      BaseClassInit(context);
    }

    #region ConvLayer

    [TestMethod]
    public void ConvLayer_Build()
    {
      var layer = simpleRectConvLayer();

      Assert.AreEqual( 3, layer.OutputHeight);
      Assert.AreEqual( 2, layer.OutputWidth);
      Assert.AreEqual(26, layer.ParamCount);
      Assert.AreEqual(26, layer.Weights.Length);

      Assert.AreEqual( 1, layer.GetKernel(0, 0, 0, 0));
      Assert.AreEqual( 0, layer.GetKernel(0, 0, 0, 1));
      Assert.AreEqual( 1, layer.GetKernel(0, 0, 1, 0));
      Assert.AreEqual(-1, layer.GetKernel(0, 0, 1, 1));
      Assert.AreEqual( 0, layer.GetKernel(0, 0, 2, 0));
      Assert.AreEqual( 1, layer.GetKernel(0, 0, 2, 1));
      Assert.AreEqual( 0, layer.GetKernel(0, 1, 0, 0));
      Assert.AreEqual( 1, layer.GetKernel(0, 1, 0, 1));
      Assert.AreEqual(-1, layer.GetKernel(0, 1, 1, 0));
      Assert.AreEqual( 2, layer.GetKernel(0, 1, 1, 1));
      Assert.AreEqual( 1, layer.GetKernel(0, 1, 2, 0));
      Assert.AreEqual(-1, layer.GetKernel(0, 1, 2, 1));
      Assert.AreEqual( 1, layer.GetBias(0));

      Assert.AreEqual(-1, layer.GetKernel(1, 0, 0, 0));
      Assert.AreEqual( 0, layer.GetKernel(1, 0, 0, 1));
      Assert.AreEqual(-1, layer.GetKernel(1, 0, 1, 0));
      Assert.AreEqual( 1, layer.GetKernel(1, 0, 1, 1));
      Assert.AreEqual(-0, layer.GetKernel(1, 0, 2, 0));
      Assert.AreEqual(-1, layer.GetKernel(1, 0, 2, 1));
      Assert.AreEqual( 0, layer.GetKernel(1, 1, 0, 0));
      Assert.AreEqual(-1, layer.GetKernel(1, 1, 0, 1));
      Assert.AreEqual( 1, layer.GetKernel(1, 1, 1, 0));
      Assert.AreEqual(-2, layer.GetKernel(1, 1, 1, 1));
      Assert.AreEqual(-1, layer.GetKernel(1, 1, 2, 0));
      Assert.AreEqual( 1, layer.GetKernel(1, 1, 2, 1));
      Assert.AreEqual(-1, layer.GetBias(1));
    }

    [TestMethod]
    public void ConvLayer_Calculate()
    {
      var layer = simpleRectConvLayer();
      var input = new double[2][,]
      {
        new double[3,4] { {1, 1, -1, -1},
                          {1, 1, -1, -1},
                          {1, 1, -1, -1}, },
        new double[3,4] { {1, 1, -1, -1},
                          {1, 1, -1, -1},
                          {1, 1, -1, -1}, }
      };
      var result = new double[2][,] { new double[3,2], new double[3,2] };

      layer.Calculate(input, result);

      Assert.AreEqual( 3, result[0][0,0]);
      Assert.AreEqual(-1, result[0][0,1]);
      Assert.AreEqual( 5, result[0][1,0]);
      Assert.AreEqual(-3, result[0][1,1]);
      Assert.AreEqual( 4, result[0][2,0]);
      Assert.AreEqual(-2, result[0][2,1]);
      Assert.AreEqual(-3, result[1][0,0]);
      Assert.AreEqual( 1, result[1][0,1]);
      Assert.AreEqual(-5, result[1][1,0]);
      Assert.AreEqual( 3, result[1][1,1]);
      Assert.AreEqual(-4, result[1][2,0]);
      Assert.AreEqual( 2, result[1][2,1]);
    }

    [TestMethod]
    public void ConvLayer_Backprop()
    {
      var layer = simpleRectConvLayer();
      var prevLayer = new LayerMock(Activation.ReLU);

      var prevValues = new double[2][,]
      {
        new double[3,4] { {1, 1, -1, -1},
                          {1, 1, -1, -1},
                          {1, 1, -1, -1}, },
        new double[3,4] { {1, 1, -1, -1},
                          {1, 1, -1, -1},
                          {1, 1, -1, -1}, }
      };
      var prevErrors = new double[2][,]
      {
        new double[3,4] { {1, 0, -1, 0},
                          {1, 0, -1, 0},
                          {1, 0, -1, 0}, },
        new double[3,4] { {1, 0, -1, 0},
                          {1, 0, -1, 0},
                          {1, 0, -1, 0}, }
      };
      var errors = new double[2][,] { new double[3,2], new double[3,2] };

      layer.Backprop(prevLayer, prevValues, prevErrors, errors);

      //Assert.AreEqual(, prevErrors[0][0,0]);
      //Assert.AreEqual(, prevErrors[0][0,1]);
      //Assert.AreEqual(, prevErrors[0][0,2]);
      //Assert.AreEqual(, prevErrors[0][0,3]);
      //Assert.AreEqual(, prevErrors[0][1,0]);
      //Assert.AreEqual(, prevErrors[0][1,1]);
      //Assert.AreEqual(, prevErrors[0][1,2]);
      //Assert.AreEqual(, prevErrors[0][1,3]);
      //Assert.AreEqual(, prevErrors[0][2,0]);
      //Assert.AreEqual(, prevErrors[0][2,1]);
      //Assert.AreEqual(, prevErrors[0][2,2]);
      //Assert.AreEqual(, prevErrors[0][2,3]);
      //
      //Assert.AreEqual(, prevErrors[1][0,0]);
      //Assert.AreEqual(, prevErrors[1][0,1]);
      //Assert.AreEqual(, prevErrors[1][0,2]);
      //Assert.AreEqual(, prevErrors[1][0,3]);
      //Assert.AreEqual(, prevErrors[1][1,0]);
      //Assert.AreEqual(, prevErrors[1][1,1]);
      //Assert.AreEqual(, prevErrors[1][1,2]);
      //Assert.AreEqual(, prevErrors[1][1,3]);
      //Assert.AreEqual(, prevErrors[1][2,0]);
      //Assert.AreEqual(, prevErrors[1][2,1]);
      //Assert.AreEqual(, prevErrors[1][2,2]);
      //Assert.AreEqual(, prevErrors[1][2,3]);
    }

    #endregion

    #region .pvt

    private ConvLayer simpleRectConvLayer()
    {
      var layer = new ConvLayer(outputDepth:   2,
                                windowHeight:  3, windowWidth:  2,
                                strideHeight:  1, strideWidth:  2,
                                paddingHeight: 1, paddingWidth: 0,
                                activation:    Activation.Identity);

      layer.InputHeight = 3;
      layer.InputWidth  = 4;
      layer.InputDepth  = 2;

      layer._Build();

      var weights = new double[]
      {
        // feature map #1
        1,  0,
        1, -1,
        0,  1,

        0,  1,
       -1,  2,
        1, -1,
        1, // bias #1

        // feature map #2
       -1,  0,
       -1,  1,
        0, -1,

        0, -1,
        1, -2,
       -1,  1,
       -1, // bias #2
      };

      Array.Copy(weights, layer.Weights, layer.Weights.Length);

      return layer;
    }


    #endregion
  }
}
