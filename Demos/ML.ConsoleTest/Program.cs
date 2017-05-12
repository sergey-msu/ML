﻿using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using ML.Core;
using ML.Core.Mathematics;
using ML.Core.ComputingNetworks;
using System.Threading;
using System.Diagnostics;

namespace ML.ConsoleTest
{
  class Program
  {
    static readonly RandomGenerator m_Generator = new RandomGenerator();

    static void Main(string[] args)
    {
      //generateNormal2Classes(200, 200);
      //generateNormal3Classes(100, 100, 100);
      //generateFlower(100);

      //var file = "primitive.csv";
      //var file = "iris.csv";
      //var file = "iris.trunk.2d.csv";
      //var file = "normal.3classes.100.csv";
      //var file = "primitive.3classes.csv";
      //var file = "normal.2classes.1000.csv";
      //var file = "normal.2classes.200.all.csv";
      //var file = "normal.2classes.200.csv";
      //var file = "normal.3classes.1000.csv";
      //var file = "normal.3classes.1000.all.csv";
      var file = "flower.3classes.100.all.csv";
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

    private static void generateFlower(int n)
    {
      using (var file = File.Open(string.Format("flower.3classes.{0}.all.csv", n), FileMode.Create))
      using (var writer = new StreamWriter(file))
      {
        writer.WriteLine("f1,f2,_class,_value,_training");

        for (int i = 0; i < n; i++)
        {
          var p = m_Generator.GenerateNormalPoint(2, 1, 0.2);
          writer.WriteLine("{0},{1},{2},{3},{4}", Math.Round(p.X, 4), Math.Round(p.Y, 4), "Green", 1, 1);
        }

        for (int i = 0; i < n; i++)
        {
          var p = m_Generator.GenerateNormalPoint(1.5, 0, 0.2);
          writer.WriteLine("{0},{1},{2},{3},{4}", Math.Round(p.X, 4), Math.Round(p.Y, 4), "Red", 2, 1);
        }
        for (int i = 0; i < n; i++)
        {
          var p = m_Generator.GenerateNormalPoint(1.5, 2.0, 0.2);
          writer.WriteLine("{0},{1},{2},{3},{4}", Math.Round(p.X, 4), Math.Round(p.Y, 4), "Red", 2, 1);
        }
        for (int i = 0; i < n; i++)
        {
          var p = m_Generator.GenerateNormalPoint(3.0, 1.0, 0.2);
          writer.WriteLine("{0},{1},{2},{3},{4}", Math.Round(p.X, 4), Math.Round(p.Y, 4), "Red", 2, 1);
        }

        for (int i = 0; i < n; i++)
        {
          var p = m_Generator.GenerateNormalPoint(1.0, 1.0, 0.2);
          writer.WriteLine("{0},{1},{2},{3},{4}", Math.Round(p.X, 4), Math.Round(p.Y, 4), "Blue", 3, 1);
        }
        for (int i = 0; i < n; i++)
        {
          var p = m_Generator.GenerateNormalPoint(2.5, 2.0, 0.2);
          writer.WriteLine("{0},{1},{2},{3},{4}", Math.Round(p.X, 4), Math.Round(p.Y, 4), "Blue", 3, 1);
        }
        for (int i = 0; i < n; i++)
        {
          var p = m_Generator.GenerateNormalPoint(2.5, 0, 0.2);
          writer.WriteLine("{0},{1},{2},{3},{4}", Math.Round(p.X, 4), Math.Round(p.Y, 4), "Blue", 3, 1);
        }


      }
    }
  }

  #region Test Canvas



  #endregion
}
