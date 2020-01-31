%
% superclass for all Coiflet wavelet parameters
%
%   "The Coiflet filters are designed to give both the mother and father
%	 wavelets 2*Par vanishing moments; here Par may be one of
%    1,2,3,4 or 5." (see [1])
%
% REFERENCES:
%   [1] WaveLab Version 850 ()
%
% author: Martin F. Schiffner
% date: 2020-01-27
% modified: 2020-01-28
%
classdef coiflet < linear_transforms.wavelets.type

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        parameter ( 1, 1 ) double { mustBeMember( parameter, [ 1, 2, 3, 4, 5 ] ) } = 5 % number of vanishing moments over 2

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = coiflet( parameters )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % property validation function ensures valid parameters

            %--------------------------------------------------------------
            % 2.) create Coiflet wavelet parameters
            %--------------------------------------------------------------
            % constructor of superclass
            objects@linear_transforms.wavelets.type( size( parameters ) );

            % iterate Coiflet wavelet parameters
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).parameter = parameters( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = coiflet( parameters )

        %------------------------------------------------------------------
        % generate orthonormal quadrature mirror filters (QMFs)
        %------------------------------------------------------------------
        function QMFs = MakeONFilter( coiflets )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.wavelets.coiflet
            if ~isa( coiflets, 'linear_transforms.wavelets.coiflet' )
                errorStruct.message = 'coiflets must be linear_transforms.wavelets.coiflet!';
                errorStruct.identifier = 'MakeONFilter:NoCoifletWavelets';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) generate orthonormal QMF
            %--------------------------------------------------------------
            % specify cell array for QMFs
            QMFs = cell( size( coiflets ) );

            % iterate Coiflet wavelet parameters
            for index_object = 1:numel( coiflets )

                % call MakeONFilter of WaveLab
                QMFs{ index_object } = MakeONFilter( 'Coiflet', coiflets( index_object ).parameter );

            end % for index_object = 1:numel( coiflets )

            % avoid cell array for single coiflets
            if isscalar( coiflets )
                QMFs = QMFs{ 1 };
            end

        end % function QMFs = MakeONFilter( coiflets )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( coiflets )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.wavelets.coiflet
            if ~isa( coiflets, 'linear_transforms.wavelets.coiflet' )
                errorStruct.message = 'coiflets must be linear_transforms.wavelets.coiflet!';
                errorStruct.identifier = 'string:NoCoifletWavelets';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) string array
            %--------------------------------------------------------------
            % initialize string array
            strs_out = repmat( "", size( coiflets ) );

            % iterate Daubechies wavelet parameters
            for index_object = 1:numel( coiflets )

                strs_out( index_object ) = sprintf( "Coiflet-%d", coiflets( index_object ).parameter );

            end % for index_object = 1:numel( coiflets )

        end % function strs_out = string( coiflets )

	end % methods

end % classdef coiflet < linear_transforms.wavelets.type
