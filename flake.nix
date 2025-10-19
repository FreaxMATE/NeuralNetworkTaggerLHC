{
  description = "mkdocs";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
  };

  outputs = { self , nixpkgs ,... }:
  let
    system = "x86_64-linux";
  in {
    devShells."${system}".default =
    let
      pkgs = import nixpkgs {
        inherit system;
      };
    in pkgs.mkShell {
      #LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
      packages = with pkgs; [
        python312Packages.python
        python312Packages.numpy
        python312Packages.matplotlib
        python312Packages.pandas
        python312Packages.scipy
        python312Packages.notebook
        python312Packages.jupyter
        python312Packages.pytorch
	python312Packages.scikit-learn
      ];
    };
  };
}
