%
% superclass for all Haar wavelet parameters
%
%	"The Haar filter (which could be considered a Daubechies-2) was the
%	 first wavelet, though not called as such, and is discontinuous."
%    (see [1])
%
% REFERENCES:
%	[1] Orthogonal/MakeONFilter in WaveLab 850 (http://www-stat.stanford.edu/~wavelab)
%
% author: Martin F. Schiffner
% date: 2020-01-27
% modified: 2020-07-14
%
classdef haar < linear_transforms.wavelets.type

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = haar( size )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty size
            if nargin < 1 || isempty( size )
                size = 1;
            end

            % superclass ensures row vector for size
            % superclass ensures nonempty positive integers for size

            %--------------------------------------------------------------
            % 2.) create Haar wavelet parameters
            %--------------------------------------------------------------
            % constructor of superclass
            objects@linear_transforms.wavelets.type( size );

        end % function objects = haar( size )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( haars )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure one argument
            narginchk( 1, 1 );

            % ensure class linear_transforms.wavelets.haar
            if ~isa( haars, 'linear_transforms.wavelets.haar' )
                errorStruct.message = 'haars must be linear_transforms.wavelets.haar!';
                errorStruct.identifier = 'string:NoHaarWavelets';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) string array
            %--------------------------------------------------------------
            % initialize string array
            strs_out = repmat( "Haar", size( haars ) );

        end % function strs_out = string( haars )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % generate orthonormal QMF (scalar)
        %------------------------------------------------------------------
        function QMF = MakeONFilter_scalar( haar )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class linear_transforms.wavelets.type (scalar) for haar

            % ensure class linear_transforms.wavelets.haar
            if ~isa( haar, 'linear_transforms.wavelets.haar' )
                errorStruct.message = 'haar must be linear_transforms.wavelets.haar!';
                errorStruct.identifier = 'MakeONFilter_scalar:NoHaarWavelet';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) generate orthonormal QMF (scalar)
            %--------------------------------------------------------------
            % call MakeONFilter of WaveLab
            QMF = MakeONFilter( 'Haar' );

        end % function QMF = MakeONFilter_scalar( haar )

	end % methods (Access = protected, Hidden)

end % classdef haar < linear_transforms.wavelets.type
