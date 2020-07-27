%
% superclass for all Vaidyanathan wavelet parameters
%
%   "The Vaidyanathan filter gives an exact reconstruction, but does not
%	 satisfy any moment condition.  The filter has been optimized for
%	 speech coding." (see [1])
%
% REFERENCES:
%	[1] Orthogonal/MakeONFilter in WaveLab 850 (http://www-stat.stanford.edu/~wavelab)
%
% author: Martin F. Schiffner
% date: 2020-01-27
% modified: 2020-07-14
%
classdef vaidyanathan < linear_transforms.wavelets.type

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = vaidyanathan( size )

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
            % 2.) create Vaidyanathan wavelet parameters
            %--------------------------------------------------------------
            % constructor of superclass
            objects@linear_transforms.wavelets.type( size );

        end % function objects = vaidyanathan( varargin )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( vaidyanathans )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure one argument
            narginchk( 1, 1 );

            % ensure class linear_transforms.wavelets.vaidyanathan
            if ~isa( vaidyanathans, 'linear_transforms.wavelets.vaidyanathan' )
                errorStruct.message = 'vaidyanathans must be linear_transforms.wavelets.vaidyanathan!';
                errorStruct.identifier = 'string:NoVaidyanathanWavelets';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) string array
            %--------------------------------------------------------------
            % initialize string array
            strs_out = repmat( "Vaidyanathan", size( vaidyanathans ) );

        end % function strs_out = string( vaidyanathans )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % generate orthonormal QMF (scalar)
        %------------------------------------------------------------------
        function QMF = MakeONFilter_scalar( vaidyanathan )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class linear_transforms.wavelets.type (scalar) for vaidyanathan

            % ensure class linear_transforms.wavelets.vaidyanathan
            if ~isa( vaidyanathan, 'linear_transforms.wavelets.vaidyanathan' )
                errorStruct.message = 'vaidyanathan must be linear_transforms.wavelets.vaidyanathan!';
                errorStruct.identifier = 'MakeONFilter_scalar:NoVaidyanathanWavelet';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) generate orthonormal QMF (scalar)
            %--------------------------------------------------------------
            % call MakeONFilter of WaveLab
            QMF = MakeONFilter( 'Vaidyanathan' );

        end % function QMF = MakeONFilter_scalar( vaidyanathan )

	end % methods (Access = protected, Hidden)

end % classdef vaidyanathan < linear_transforms.wavelets.type
