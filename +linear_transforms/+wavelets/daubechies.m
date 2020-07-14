%
% superclass for all Daubechies wavelet parameters
%
%   "The Daubechies filters are minimal phase filters that generate wavelets
%	 which have a minimal support for a given number of vanishing moments.
%	 They are indexed by their length, Par, which may be one of
%	 4,6,8,10,12,14,16,18 or 20. The number of vanishing moments is par/2."
%    (see [1])
%
% REFERENCES:
%	[1] Orthogonal/MakeONFilter in WaveLab 850 (http://www-stat.stanford.edu/~wavelab)
%
% author: Martin F. Schiffner
% date: 2020-01-27
% modified: 2020-07-14
%
classdef daubechies < linear_transforms.wavelets.type

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        length ( 1, 1 ) double { mustBeMember( length, [ 4, 6, 8, 10, 12, 14, 16, 18, 20 ] ) } = 20 % length / number of vanishing moments over 2

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = daubechies( lengths )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure at most one argument
            narginchk( 0, 1 );

            % ensure nonempty size
            if nargin < 1 || isempty( lengths )
                lengths = 20;
            end

            % property validation function ensures valid lengths

            %--------------------------------------------------------------
            % 2.) create Daubechies wavelet parameters
            %--------------------------------------------------------------
            % constructor of superclass
            objects@linear_transforms.wavelets.type( size( lengths ) );

            % iterate Daubechies wavelet parameters
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).length = lengths( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = daubechies( lengths )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( daubechies )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure one argument
            narginchk( 1, 1 );

            % ensure class linear_transforms.wavelets.daubechies
            if ~isa( daubechies, 'linear_transforms.wavelets.daubechies' )
                errorStruct.message = 'daubechies must be linear_transforms.wavelets.daubechies!';
                errorStruct.identifier = 'string:NoDaubechiesWavelets';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) string array
            %--------------------------------------------------------------
            % initialize string array
            strs_out = repmat( "", size( daubechies ) );

            % iterate Daubechies wavelet parameters
            for index_object = 1:numel( daubechies )

                strs_out( index_object ) = sprintf( "Daubechies-%d", daubechies( index_object ).length );

            end % for index_object = 1:numel( daubechies )

        end % function strs_out = string( daubechies )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % generate orthonormal QMF (scalar)
        %------------------------------------------------------------------
        function QMF = MakeONFilter_scalar( daubechies )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class linear_transforms.wavelets.type (scalar) for daubechies

            % ensure class linear_transforms.wavelets.daubechies
            if ~isa( daubechies, 'linear_transforms.wavelets.daubechies' )
                errorStruct.message = 'daubechies must be linear_transforms.wavelets.daubechies!';
                errorStruct.identifier = 'MakeONFilter_scalar:NoDaubechiesWavelet';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) generate orthonormal QMF (scalar)
            %--------------------------------------------------------------
            % call MakeONFilter of WaveLab
            QMF = MakeONFilter( 'Daubechies', daubechies.length );

        end % function QMF = MakeONFilter_scalar( daubechies )

	end % methods (Access = protected, Hidden)

end % classdef daubechies < linear_transforms.wavelets.type
