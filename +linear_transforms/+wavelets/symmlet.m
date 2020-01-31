%
% superclass for all Symmlet wavelet parameters
%
%	"Symmlets are also wavelets within a minimum size support for a given
%	 number of vanishing moments, but they are as symmetrical as possible,
%	 as opposed to the Daubechies filters which are highly asymmetrical.
%	 They are indexed by Par, which specifies the number of vanishing
%	 moments and is equal to half the size of the support. It ranges
%	 from 4 to 10." (see [1])
%
% REFERENCES:
%   [1] WaveLab Version 850 ()
%
% author: Martin F. Schiffner
% date: 2020-01-27
% modified: 2020-01-28
%
classdef symmlet < linear_transforms.wavelets.type

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        parameter ( 1, 1 ) double { mustBeMember( parameter, [ 4, 5, 6, 7, 8, 9, 10 ] ) } = 10 % number of vanishing moments / half the size of the support

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = symmlet( parameters )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % property validation function ensures valid parameters

            %--------------------------------------------------------------
            % 2.) create Symmlet wavelet parameters
            %--------------------------------------------------------------
            % constructor of superclass
            objects@linear_transforms.wavelets.type( size( parameters ) );

            % iterate Symmlet wavelet parameters
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).parameter = parameters( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = symmlet( parameters )

        %------------------------------------------------------------------
        % generate orthonormal quadrature mirror filters (QMFs)
        %------------------------------------------------------------------
        function QMFs = MakeONFilter( symmlets )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.wavelets.symmlet
            if ~isa( symmlets, 'linear_transforms.wavelets.symmlet' )
                errorStruct.message = 'symmlets must be linear_transforms.wavelets.symmlet!';
                errorStruct.identifier = 'MakeONFilter:NoSymmletWavelets';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) generate orthonormal QMF
            %--------------------------------------------------------------
            % specify cell array for QMFs
            QMFs = cell( size( symmlets ) );

            % iterate Symmlet wavelet parameters
            for index_object = 1:numel( symmlets )

                % call MakeONFilter of WaveLab
                QMFs{ index_object } = MakeONFilter( 'Symmlet', symmlets( index_object ).parameter );

            end % for index_object = 1:numel( symmlets )

            % avoid cell array for single symmlets
            if isscalar( symmlets )
                QMFs = QMFs{ 1 };
            end

        end % function QMFs = MakeONFilter( symmlets )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( symmlets )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.wavelets.symmlet
            if ~isa( symmlets, 'linear_transforms.wavelets.symmlet' )
                errorStruct.message = 'symmlets must be linear_transforms.wavelets.symmlet!';
                errorStruct.identifier = 'string:NoSymmletWavelets';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) string array
            %--------------------------------------------------------------
            % initialize string array
            strs_out = repmat( "", size( symmlets ) );

            % iterate Daubechies wavelet parameters
            for index_object = 1:numel( symmlets )

                strs_out( index_object ) = sprintf( "Symmlet-%d", symmlets( index_object ).parameter );

            end % for index_object = 1:numel( symmlets )

        end % function strs_out = string( symmlets )

	end % methods

end % classdef symmlet < linear_transforms.wavelets.type
