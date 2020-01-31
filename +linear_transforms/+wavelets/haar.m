%
% superclass for all Haar wavelet parameters
%
%	"The Haar filter (which could be considered a Daubechies-2) was the
%	 first wavelet, though not called as such, and is discontinuous."
%    (see [1])
%
% REFERENCES:
%   [1] WaveLab Version 850 ()
%
% author: Martin F. Schiffner
% date: 2020-01-27
% modified: 2020-01-28
%
classdef haar < linear_transforms.wavelets.type

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = haar( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty size
            if nargin >= 1 && ~isempty( varargin{ 1 } )
                size = varargin{ 1 };
            else
                size = 1;
            end

            % superclass ensures row vector for size
            % superclass ensures nonempty positive integers for size

            %--------------------------------------------------------------
            % 2.) create Haar wavelet parameters
            %--------------------------------------------------------------
            % constructor of superclass
            objects@linear_transforms.wavelets.type( size );

        end % function objects = haar( varargin )

        %------------------------------------------------------------------
        % generate orthonormal quadrature mirror filters (QMFs)
        %------------------------------------------------------------------
        function QMFs = MakeONFilter( haars )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.wavelets.haar
            if ~isa( haars, 'linear_transforms.wavelets.haar' )
                errorStruct.message = 'haars must be linear_transforms.wavelets.haar!';
                errorStruct.identifier = 'MakeONFilter:NoHaarWavelets';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) generate orthonormal QMF
            %--------------------------------------------------------------
            % specify cell array for QMFs
            QMFs = cell( size( haars ) );

            % iterate Daubechies wavelet parameters
            for index_object = 1:numel( haars )

                % call MakeONFilter of WaveLab
                QMFs{ index_object } = MakeONFilter( 'Haar' );

            end % for index_object = 1:numel( haars )

            % avoid cell array for single haars
            if isscalar( haars )
                QMFs = QMFs{ 1 };
            end

        end % function QMFs = MakeONFilter( haars )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( haars )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
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

end % classdef haar < linear_transforms.wavelets.type
