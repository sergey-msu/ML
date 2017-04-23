using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ML.DeepMethods.Optimizers;

namespace ML.Tests.UnitTests.CNN
{
  [TestClass]
  public class OptimizerTests : TestBase
  {
    [TestMethod]
    public void NopeOptimizer_Push()
    {
      // arrange
      var weights = new double[3][]
      {
        new[] { 1.0D, 2.0D, -1.0D },
        null,
        new[] { 0.0D, 1.0D }
      };
      var gradient = new double[3][]
      {
        new[] { 1.5D, -1.0D, -2.5D },
        null,
        new[] { 1.0D, 0.5D }
      };
      var lr = 2.0D;
      var optimizer = new SGDOptimizer();

      // act
      optimizer.Init(weights);
      optimizer.Push(gradient, lr);

      // assert
      Assert.AreEqual(-2, weights[0][0]);
      Assert.AreEqual( 4, weights[0][1]);
      Assert.AreEqual( 4, weights[0][2]);
      Assert.AreEqual(-2, weights[2][0]);
      Assert.AreEqual( 0, weights[2][1]);
      Assert.AreEqual(43, optimizer.Step2);
    }

    [TestMethod]
    public void NopeOptimizer_2Push()
    {
      // arrange
      var weights = new double[3][]
      {
        new[] { 1.0D, 2.0D, -1.0D },
        null,
        new[] { 0.0D, 1.0D }
      };
      var gradient1 = new double[3][]
      {
        new[] { 1.5D, -1.0D, -2.5D },
        null,
        new[] { 1.0D, 0.5D }
      };
      var gradient2 = new double[3][]
      {
        new[] { 0.5D, -1.5D, -2.0D },
        null,
        new[] { 0.0D, -0.5D }
      };
      var lr = 2.0D;
      var optimizer = new SGDOptimizer();

      // act
      optimizer.Init(weights);
      optimizer.Push(gradient1, lr);
      optimizer.Push(gradient2, lr);

      // assert
      Assert.AreEqual(-3, weights[0][0]);
      Assert.AreEqual( 7, weights[0][1]);
      Assert.AreEqual( 8, weights[0][2]);
      Assert.AreEqual(-2, weights[2][0]);
      Assert.AreEqual( 1, weights[2][1]);
      Assert.AreEqual(27, optimizer.Step2);
    }

    [TestMethod]
    public void MomentumOptimizer_Push()
    {
      // arrange
      var weights = new double[3][]
      {
        new[] { 1.0D, 2.0D, -1.0D },
        null,
        new[] { 0.0D, 1.0D }
      };
      var gradient = new double[3][]
      {
        new[] { 1.5D, -1.0D, -2.5D },
        null,
        new[] { 1.0D, 0.5D }
      };
      var lr = 2.0D;
      var mu = 0.5D;
      var optimizer = new MomentumOptimizer(mu);

      // act
      optimizer.Init(weights);
      optimizer.Push(gradient, lr);

      // assert
      Assert.AreEqual(-2, weights[0][0]);
      Assert.AreEqual( 4, weights[0][1]);
      Assert.AreEqual( 4, weights[0][2]);
      Assert.AreEqual(-2, weights[2][0]);
      Assert.AreEqual( 0, weights[2][1]);
      Assert.AreEqual(43, optimizer.Step2);
    }

    [TestMethod]
    public void MomentumOptimizer_2Push()
    {
      // arrange
      var weights = new double[3][]
      {
        new[] { 1.0D, 2.0D, -1.0D },
        null,
        new[] { 0.0D, 1.0D }
      };
      var gradient1 = new double[3][]
      {
        new[] { 1.5D, -1.0D, -2.5D },
        null,
        new[] { 1.0D, 0.5D }
      };
      var gradient2 = new double[3][]
      {
        new[] { 0.5D, -1.5D, -2.0D },
        null,
        new[] { 0.0D, -0.5D }
      };
      var lr = 2.0D;
      var mu = 0.5D;
      var optimizer = new MomentumOptimizer(mu);

      // act
      optimizer.Init(weights);
      optimizer.Push(gradient1, lr);
      optimizer.Push(gradient2, lr);

      // assert
      Assert.AreEqual(-4.5,  weights[0][0]);
      Assert.AreEqual( 8,    weights[0][1]);
      Assert.AreEqual(10.5,  weights[0][2]);
      Assert.AreEqual(-3,    weights[2][0]);
      Assert.AreEqual( 0.5,  weights[2][1]);
      Assert.AreEqual(65.75, optimizer.Step2);
    }

    [TestMethod]
    public void MomentumOptimizer_3Push()
    {
      // arrange
      var weights = new double[3][]
      {
        new[] { 1.0D, 2.0D, -1.0D },
        null,
        new[] { 0.0D, 1.0D }
      };
      var gradient1 = new double[3][]
      {
        new[] { 1.5D, -1.0D, -2.5D },
        null,
        new[] { 1.0D, 0.5D }
      };
      var gradient2 = new double[3][]
      {
        new[] { 0.5D, -1.5D, -2.0D },
        null,
        new[] { 0.0D, -0.5D }
      };
      var gradient3 = new double[3][]
      {
        new[] { 0.5D, -0.5D, 0.0D },
        null,
        new[] { 0.5D, -0.5D }
      };
      var lr = 2.0D;
      var mu = 0.5D;
      var optimizer = new MomentumOptimizer(mu);

      // act
      optimizer.Init(weights);
      optimizer.Push(gradient1, lr);
      optimizer.Push(gradient2, lr);
      optimizer.Push(gradient3, lr);

      // assert
      Assert.AreEqual(-6.75,   weights[0][0]);
      Assert.AreEqual( 11,     weights[0][1]);
      Assert.AreEqual( 13.75,  weights[0][2]);
      Assert.AreEqual( -4.5,   weights[2][0]);
      Assert.AreEqual(  1.75,  weights[2][1]);
      Assert.AreEqual(28.4375, optimizer.Step2);
    }

    [TestMethod]
    public void AdagradOptimizer_Push()
    {
      // arrange
      var weights = new double[3][]
      {
        new[] { 1.0D, 2.0D, -1.0D },
        null,
        new[] { 0.0D, 1.0D }
      };
      var gradient = new double[3][]
      {
        new[] { 1.5D, -1.0D, -2.5D },
        null,
        new[] { 1.0D, 0.5D }
      };
      var lr = 2.0D;
      var eps = 1.0D;
      var optimizer = new AdagradOptimizer(eps);

      // act
      optimizer.Init(weights);
      optimizer.Push(gradient, lr);

      // assert

      var adaLr = new double[3][]  // 1/sqrt(G + eps)
      {
        new[] { 0.554700196D, 0.70710678D, 0.37139067D },
        null,
        new[] { 0.707106781D, 0.89442719D }
      };
      var dw = new double[3][]  // -lr*adaLr*DL
      {
        new[] { -1.664100588D, 1.41421356D, 1.85695335D },
        null,
        new[] { -1.41421356D, -0.89442719D }
      };
      var step2 = dw[0][0]*dw[0][0] + dw[0][1]*dw[0][1] + dw[0][2]*dw[0][2] + dw[2][0]*dw[2][0] + dw[2][1]*dw[2][1];

      Assert.AreEqual( 1.0D+dw[0][0], weights[0][0], EPS_ROUGH);
      Assert.AreEqual( 2.0D+dw[0][1], weights[0][1], EPS_ROUGH);
      Assert.AreEqual(-1.0D+dw[0][2], weights[0][2], EPS_ROUGH);
      Assert.AreEqual( 0.0D+dw[2][0], weights[2][0], EPS_ROUGH);
      Assert.AreEqual( 1.0D+dw[2][1], weights[2][1], EPS_ROUGH);
      Assert.AreEqual(step2, optimizer.Step2, EPS_ROUGH);
    }

    [TestMethod]
    public void AdagradOptimizer_2Push()
    {
      // arrange
      var weights = new double[3][]
      {
        new[] { 1.0D, 2.0D, -1.0D },
        null,
        new[] { 0.0D, 1.0D }
      };
      var gradient1 = new double[3][]
      {
        new[] { 1.5D, -1.0D, -2.5D },
        null,
        new[] { 1.0D, 0.5D }
      };
      var gradient2 = new double[3][]
      {
        new[] { 1.0D, -1.0D, 0.0D },
        null,
        new[] { 0.0D, 1.0D }
      };
      var lr = 2.0D;
      var eps = 1.0D;
      var optimizer = new AdagradOptimizer(eps);

      // act
      optimizer.Init(weights);
      optimizer.Push(gradient1, lr);
      optimizer.Push(gradient2, lr);

      // assert

      var newWeights =  new double[3][]
      {
        new[] { -0.66410058867D, 3.414213562373D, 0.85695338177D },
        null,
        new[] { -1.41421356237D, 0.105572809000D }
      };
      var adaLr = new double[3][]  // 1/sqrt(G + eps)
      {
        new[] { 0.485071250072666D, 0.577350269189626D, 0.371390676354104D },
        null,
        new[] { 0.707106781186547D, 0.666666666666667D }
      };
      var dw = new double[3][]  // -lr*adaLr*DL
      {
        new[] { -0.9701425001453D, 1.1547005383793D, 0.0D },
        null,
        new[] { 0.0D, -1.3333333333333D }
      };
      var step2 = dw[0][0]*dw[0][0] + dw[0][1]*dw[0][1] + dw[0][2]*dw[0][2] + dw[2][0]*dw[2][0] + dw[2][1]*dw[2][1];

      Assert.AreEqual(newWeights[0][0]+dw[0][0], weights[0][0], EPS_ROUGH);
      Assert.AreEqual(newWeights[0][1]+dw[0][1], weights[0][1], EPS_ROUGH);
      Assert.AreEqual(newWeights[0][2]+dw[0][2], weights[0][2], EPS_ROUGH);
      Assert.AreEqual(newWeights[2][0]+dw[2][0], weights[2][0], EPS_ROUGH);
      Assert.AreEqual(newWeights[2][1]+dw[2][1], weights[2][1], EPS_ROUGH);
      Assert.AreEqual(step2, optimizer.Step2, EPS_ROUGH);
    }


  }
}
