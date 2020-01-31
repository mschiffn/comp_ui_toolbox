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
%   [1] WaveLab Version 850 ()
%
% author: Martin F. Schiffner
% date: 2020-01-27
% modified: 2020-01-28
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
        % generate orthonormal quadrature mirror filters (QMFs)
        %------------------------------------------------------------------
        function QMFs = MakeONFilter( daubechies )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.wavelets.daubechies
            if ~isa( daubechies, 'linear_transforms.wavelets.daubechies' )
                errorStruct.message = 'daubechies must be linear_transforms.wavelets.daubechies!';
                errorStruct.identifier = 'MakeONFilter:NoDaubechiesWavelets';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) generate orthonormal QMF
            %--------------------------------------------------------------
            % specify cell array for QMFs
            QMFs = cell( size( daubechies ) );

            % iterate Daubechies wavelet parameters
            for index_object = 1:numel( daubechies )

                % call MakeONFilter of WaveLab
                QMFs{ index_object } = MakeONFilter( 'Daubechies', daubechies( index_object ).length );

            end % for index_object = 1:numel( daubechies )

            % avoid cell array for single daubechies
            if isscalar( daubechies )
                QMFs = QMFs{ 1 };
            end

        end % function QMFs = MakeONFilter( daubechies )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( daubechies )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
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

end % classdef daubechies < linear_transforms.wavelets.type
