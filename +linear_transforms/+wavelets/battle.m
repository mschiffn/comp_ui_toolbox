%
% superclass for all Battle-Lemarie wavelet parameters
%
%    "The Battle-Lemarie filter generate[s] sic! spline orthogonal wavelet basis.
%     The parameter Par gives the degree of the spline. The number of 
%     vanishing moments is Par+1." (see [1])
%
% REFERENCES:
%   [1] WaveLab Version 850 ()
%
% author: Martin F. Schiffner
% date: 2020-01-27
% modified: 2020-01-28
%
classdef battle < linear_transforms.wavelets.type

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        degree ( 1, 1 ) double { mustBeMember( degree, [ 1, 3, 5 ] ) } = 5 % degree of the spline / number of vanishing moments minus 1 (1, 3, 5)

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = battle( degrees )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % property validation function ensures valid degrees

            %--------------------------------------------------------------
            % 2.) create Battle-Lemarie wavelet parameters
            %--------------------------------------------------------------
            % constructor of superclass
            objects@linear_transforms.wavelets.type( size( degrees ) );

            % iterate Battle-Lemarie wavelet parameters
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).degree = degrees( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = battle( degrees )

        %------------------------------------------------------------------
        % generate orthonormal quadrature mirror filters (QMFs)
        %------------------------------------------------------------------
        function QMFs = MakeONFilter( battles )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.wavelets.battle
            if ~isa( battles, 'linear_transforms.wavelets.battle' )
                errorStruct.message = 'battles must be linear_transforms.wavelets.battle!';
                errorStruct.identifier = 'MakeONFilter:NoBattleLemarieWavelets';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) generate orthonormal QMF
            %--------------------------------------------------------------
            % specify cell array for QMFs
            QMFs = cell( size( battles ) );

            % iterate Battle-Lemarie wavelet parameters
            for index_object = 1:numel( battles )

                % call MakeONFilter of WaveLab
                QMFs{ index_object } = MakeONFilter( 'Battle', battles( index_object ).degree );

            end % for index_object = 1:numel( battles )

            % avoid cell array for single battles
            if isscalar( battles )
                QMFs = QMFs{ 1 };
            end

        end % function QMFs = MakeONFilter( battles )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( battles )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.wavelets.battle
            if ~isa( battles, 'linear_transforms.wavelets.battle' )
                errorStruct.message = 'battles must be linear_transforms.wavelets.battle!';
                errorStruct.identifier = 'string:NoBattleLemarieWavelets';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) string array
            %--------------------------------------------------------------
            % initialize string array
            strs_out = repmat( "", size( battles ) );

            % iterate Daubechies wavelet parameters
            for index_object = 1:numel( battles )

                strs_out( index_object ) = sprintf( "Battle-Lemarie-%d", battles( index_object ).degree );

            end % for index_object = 1:numel( battles )

        end % function strs_out = string( battles )

	end % methods

end % classdef battle < linear_transforms.wavelets.type
