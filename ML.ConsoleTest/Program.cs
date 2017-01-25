using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;

namespace ML.ConsoleTest
{
  class Program
  {
    static void Main(string[] args)
    {
      //generateNormal2Classes(200, 200);
      //generateNormal3Classes(100, 100, 100);

      //var file = "primitive.csv";
      //var file = "iris.csv";
      //var file = "normal.3classes.100.csv";
      //var file = "normal.2classes.1000.csv";
      //var file = "normal.3classes.1000.csv";
      var file = "primitive2.csv";

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
          var p1 = ML.Mathematics.Utils.GenerateNormalPoint(0, 0, 1);
          writer.WriteLine("{0},{1},{2},{3}", Math.Round(p1.X, 4), Math.Round(p1.Y, 4), "Green", i % s1 == 0 ? 1 : 0);
        }

        var s2 = n2/30;
        for (int i = 0; i < n2; i++)
        {
          var p2 = ML.Mathematics.Utils.GenerateNormalPoint(2.5, 0, 0.5);
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
          var p1 = ML.Mathematics.Utils.GenerateNormalPoint(0, 0, 1);
          writer.WriteLine("{0},{1},{2},{3},{4}", Math.Round(p1.X, 4), Math.Round(p1.Y, 4), "Green", 1, i % s1 == 0 ? 1 : 0);
        }

        var s2 = n2/20;
        for (int i = 0; i < n2; i++)
        {
          var p2 = ML.Mathematics.Utils.GenerateNormalPoint(2.5, 0, 0.5);
          writer.WriteLine("{0},{1},{2},{3},{4}", Math.Round(p2.X, 4), Math.Round(p2.Y, 4), "Red", 2, i % s2 == 0 ? 1 : 0);
        }

        var s3 = n3/20;
        for (int i = 0; i < n3; i++)
        {
          var p3 = ML.Mathematics.Utils.GenerateNormalPoint(1.7, 1.8, 0.5);
          writer.WriteLine("{0},{1},{2},{3},{4}", Math.Round(p3.X, 4), Math.Round(p3.Y, 4), "Blue", 3, i % s3 == 0 ? 1 : 0);
        }
      }
    }

  }
}
