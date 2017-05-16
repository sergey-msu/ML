﻿using System;

namespace ML.DeepTests
{
  class Program
  {
    static void Main(string[] args)
    {
      var runner =
        //new OriginalMNIST();
        //new OriginalCIFAR10();
        //new OriginalCIFAR10Trunc();
        //new KaggleMNIST();
        //new KaggleCIFAR10();
        //new KaggleCatDog();
        //new KaggleCatDogTrunk();
        new KaggleCatDog64Trunk();

      try
      {
        Console.WriteLine("STARTED at {0}", DateTime.Now);
        runner.Run();
        Console.WriteLine("DONE at {0}", DateTime.Now);
      }
      catch (Exception ex)
      {
        Console.WriteLine("ERROR:");
        Console.WriteLine(ex);

        Utils.SaveAlgCrushResults(runner.Alg, runner.OutputPath);
      }

      Console.ReadLine();
    }
  }
}
